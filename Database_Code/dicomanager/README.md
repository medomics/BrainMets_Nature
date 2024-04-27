DICOManager adapted to the analysis of brain metastases

To rebuild the database and recompute the statistics on your own data:
1. Create a python environment and install the dependencies.yml file.
2. Create a directory with all of the DICOM files. These files do not have to be sorted or organized in any particular way.
3. Open up `./build.py`
4. On `line 13` specify the directory where the database and all processed data should be stored.
5. On `line 19` specify an institution name for the dataset, if desired.
6. On `line 21` specify if you want to build a new database from scratch, or add to an existing database.
7. Download any atlas files to calculate statistics with.
8. Run `./build.py` and ingest any atlases with their institution specified. Add that institution to the atlases that will be used on `line 49`.
9. Run `./build.py` and wait for it to construct the database, digest the data, assemble the images and structures, and calculate statistics.