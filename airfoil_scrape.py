"""
Script to scrape airfoil profiles from http://airfoiltools.com/ with properties
for Re=200,000 and Ncrit=9.
Assumptions:
 - profile definition start at TE/SS and end at TE/PS and are continuous (Selig)
 - Consistent runs of profile simulations through XFoil

Run:
 > python3 airfoil_scrape.py --points
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import os
import argparse
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import ray
import more_itertools as mit

################################################################################
# %% COMMAND LINE ARGUMENT PARSING
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("POINTS", help="number of points around circumference")
parser.add_argument("CPUS", help="number of CPUs to use for ray")
parser.add_argument("--cosine", help="use cosine spacing to refine LE", action="store_true")
args = parser.parse_args()

################################################################################
# %% CONSTANTS
################################################################################

BASE_URL = "http://airfoiltools.com"
POINTS = int(args.POINTS)
CPUS = int(args.CPUS)

################################################################################
# %% CREATE FOLDER IF NOT YET EXISTING
################################################################################

try:
    os.stat(f'01-prep-data/{POINTS}')
except:
    os.mkdir(f'01-prep-data/{POINTS}')

################################################################################
# %% INIT
################################################################################

ray.init(num_cpus=CPUS)

################################################################################
# %% SCAN FOR AIRFOILS
################################################################################

##### GRAB OVERVIEW PAGE
page = requests.get(BASE_URL+"/search/airfoils/index.html")
overview_soup = BeautifulSoup(page.text, 'html.parser')

##### FIND ALL LINKS
links = overview_soup.find_all("a", href=re.compile("airfoil/details"))
print(f'Found {len(links)} airfoil definitions')
links = [link.get('href') for link in links]

##### SPLIT INTO CPU-SIZED CHUNKS
chunk_links = [list(c) for c in mit.divide(CPUS, links)]

##### DEF REMOTE FUNCTION HERE!
@ray.remote
def scrape(links, BASE_URL):

    ##### INIT BATCH ARRAYS
    X_batch = np.zeros((0, POINTS, 2, 1), dtype=float)
    y_batch = np.zeros((0, 3), dtype=float)

    ##### LOOP OVER LINKS
    for link in links:

        ##### GRAB PROFILE PAGE
        page = requests.get(BASE_URL+link)
        airfoil_soup = BeautifulSoup(page.text, 'html.parser')

        ##### FIND DAT FILE CONTAINING PROFILE
        dat_link = airfoil_soup.find("a", href=re.compile("seligdat"))

        ##### GRAB PROFILE
        profile_page = requests.get(BASE_URL+dat_link.get('href'))
        profile_soup = BeautifulSoup(profile_page.text, 'html.parser')

        ##### PARSE PROFILE
        text = profile_soup.text.split('\n')
        profile_name = text[0]
        raw_profile = np.loadtxt(text[1:])

        ##### INTERPOLATE TO POINTS
        # RAW PROFILE
        fp1 = raw_profile[:, 0]
        fp2 = raw_profile[:, 1]

        # SCALE TO LOCAL DISTANCE
        delta = np.sqrt(np.diff(fp1)**2 + np.diff(fp2)**2)
        xp = delta.cumsum()
        xp = np.insert(xp, 0, 0)/xp[-1]

        # TARGET DISTRIBUTION
        if not args.cosine:
            ##### LINEAR
            x = np.linspace(0, 1, POINTS)
        elif args.cosine:
            ##### COSINE
            x = np.linspace(0,1, POINTS) + 0.1*np.sin(np.linspace(0, 2*np.pi, POINTS))

        # INTERPOLATE
        base_profile = np.array([np.interp(x, xp, fp1), np.interp(x, xp, fp2)]).T

        ##### SCALE BACK TO 0-1
        xmin = base_profile[:, 0].min()
        xmax = base_profile[:, 0].max()
        base_profile[:, 0] = (base_profile[:, 0]-xmin)/(xmax-xmin)
        base_profile[:, 1] = (base_profile[:, 1])/(xmax-xmin)

        ##### CALCULATE AREA
        area = 0.0
        for i in range(len(base_profile)-1):
            area += 0.5*(base_profile[i, 0]-base_profile[i+1, 0])*(base_profile[i, 1]+base_profile[i+1, 1])

        ##### FIND CL/CD DATA LINK
        data_link = airfoil_soup.find_all("a", href=re.compile("^/polar/details.*200000$"))

        # FOLLOW IF RE 200k DATA EXISTS AND AREA REASONABLE
        if data_link and area <= 0.2:
            data_page = requests.get(BASE_URL+data_link[0].get('href'))
            data_soup = BeautifulSoup(data_page.text, 'html.parser')

            ##### PARSE DATA TABLE
            data = []
            table = data_soup.find('table', attrs={'class':'tabdata'})
            rows = table.find_all('tr')
            for row in rows[1:]:
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                data.append([ele for ele in cols if ele])
            data = np.array(data).astype(float)[:, 0:3]

            ##### INIT SAMPLE ARRAY
            X = np.zeros((len(data), POINTS, 2, 1), dtype=float)
            y = np.zeros((len(data), 3), dtype=float)

            ##### RUN THROUGH ANGLES AND BUILD PROFILES
            for sample in range(len(data)):

                ##### ANLGE IN RADIANS
                alpha = -data[sample, 0]*np.pi/180.0

                shift_profile = base_profile.copy()
                shift_profile[:, 0] = shift_profile[:, 0] - 0.5

                ##### ROTATE!
                profile_x = shift_profile[:, 0]*np.cos(alpha) - shift_profile[:, 1]*np.sin(alpha)
                profile_y = shift_profile[:, 0]*np.sin(alpha) + shift_profile[:, 1]*np.cos(alpha)

                ##### COMBINE
                profile_x = profile_x + 0.5
                profile = np.concatenate((profile_x.reshape(-1, 1), profile_y.reshape(-1, 1)), axis=1)

                ##### SET SAMPLE
                X[sample, :, :, 0] = profile
                y[sample, 0:2] = data[sample, 1:3]
                y[sample, 2] = area


            ##### COMBINE
            X_batch = np.concatenate((X_batch, X), axis=0)
            y_batch = np.concatenate((y_batch, y), axis=0)

    ##### RETURN!
    return X_batch, y_batch

################################################################################
# %% RUN IN PARALLEL!
################################################################################

object_id = [scrape.remote(chunk_links[batch], BASE_URL) for batch in range(CPUS)]
out = ray.get(object_id)

################################################################################
# %% SHUFFLE AND SAVE FILES
################################################################################

for i, batch in enumerate(out):

    ##### GET LOCAL BATCH
    X_batch = batch[0]
    y_batch = batch[1]

    ##### SHUFFLE
    idx = np.random.permutation(len(X_batch))
    X_batch = X_batch[idx]
    y_batch = y_batch[idx]

    ##### SAVE
    np.save(f'01-prep-data/{POINTS}/X_{i:03d}.npy', X_batch)
    np.save(f'01-prep-data/{POINTS}/y_{i:03d}.npy', y_batch)
