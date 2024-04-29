import dbutils as dbu
import pandas as pd
import utils
import psutil
import assembly
import disassembly
import contextlib
import dcmio
import statistics
import augmentations
from tqdm import tqdm
import numpy as np
from multiprocess import Pool
from datetime import datetime
pd.set_option("max_colwidth", 100)


class Cohort:
    """Assembles a Cohort data group of dicoms, from which filtering, assembly and augmentation can be applied
    """
    def __init__(self, dicom_dir: str = None,
                 db_name: str = 'file::memory:?cache=shared',
                 config_file: str = 'table_config.yaml',
                 filter_by: str = 'filter_by.yaml',
                 njobs: int = None,
                 logfile: str = 'DICOManager.log',
                 progress_bar: bool = True,
                 copy_dir: str = None,
                 only_digest_new: bool = False,
                 institution: str = None) -> None:
        """Initalizes a cohort data group. If a dicom directory is specified, the dicoms will be digested
            into a SQLite database at the specified db_name or default to in-memory, as defined by the
            config_file. Dicom digestion will be regulated by filter_by. All processes run within cohort
            will spool of n_jobs, defaulting to the number of logical CPU cores present on the machine.

        Args:
            dicom_dir (str, optional): A directory containing DICOMs to add to the cohort database. Defaults to None.
            db_name (_type_, optional): The name of the database on disk, if not specified, an in-memory database
                will be used instead. Defaults to 'file::memory:?cache=shared'.
            config_file (str, optional): A configuration of the database tables. Defaults to 'table_config.yaml'.
            filter_by (str, optional): A yaml file specifying which DICOMs to digest into the
                database. Defaults to 'filter_by.yaml.
            njobs (int, optional): Number of parallel jobs to allow the cohort object ot run. Defaults to the
                number of logical system cores.
            logfile (str, optional): Location to write any warnings or errors during assembly. Defaults to
                writing the log to './DICOManager.log'.
            progress_bar (bool, optional): Display progress bar during major operations for this cohort. These
                operations include creation, assembly, file digestion and saving. Defaults to True.
            copy_dir (str, optional): Rewrites the DICOMs to a new directory location if specified. Defaults to None.
            only_digest_new (bool, optional): Only digests new files, as indicated by the filepath. If the copy_dir
                parameter is True, this will need to open all files to determine if it has already been digested. It
                is recommended if this is the case to carefully manage the pre-existing DICOMs to reduce workload.
                Defaults to False.
        """
        self._conn = dbu.initialize_db(db_name)
        self.db_name = db_name
        self.config_file = config_file
        self.dicom_dir = dicom_dir
        self.logfile = logfile
        self.progress_bar = progress_bar
        self.rtstruct_creator = disassembly.StructCreator(self.db_name, self.logfile)
        self._logger = utils.make_log(logfile, db_name)
        self._log_n_lines = utils.line_count(logfile)

        if njobs:
            self.njobs = njobs
        else:
            self.njobs = psutil.cpu_count(logical=False)

        if dicom_dir:
            self.add_dicoms(self.dicom_dir, filter_by, progress_bar=self.progress_bar, copy_dir=copy_dir,
                            only_digest_new=only_digest_new, institution=institution)

    def _new_df(self, columns, modalities):
        column_names = []
        for mod in modalities:
            for field in columns:
                column_names.append(mod + ' ' + field)
        return pd.DataFrame(columns=column_names)

    def _to_df(self, query_call, columns):
        with contextlib.closing(dbu.make_connection(self.db_name)) as conn:
            with conn:
                df_query = pd.read_sql_query(query_call, conn)
                df_modalities = pd.read_sql_query("SELECT DISTINCT(Modality) FROM dicoms", conn)
                df_out = self._new_df(columns, df_modalities['Modality'])

        for row in df_query.itertuples():
            for field in row._fields:
                if field in ['Index', 'PatientID', 'Modality']:
                    continue
                if field == 'dates':
                    dates = list(set(row.dates.split(',')))
                    try:
                        dt_dates = [datetime.strptime(x, '%Y%m%d') for x in dates]
                    except Exception:
                        dt_dates = None
                    df_out.loc[row.PatientID, row.Modality + ' ' + field] = dt_dates
                else:
                    df_out.loc[row.PatientID, row.Modality + ' ' + field] = getattr(row, field)

        df_out.reset_index(inplace=True)
        df_out.rename(columns={'index': 'PatientID'}, inplace=True)
        return df_out

    def add_dicoms(self, dicom_dir, filter_by: str = 'filter_by.yaml', njobs: int = None, progress_bar: bool = True,
                   copy_dir: str = None, only_digest_new: bool = False, institution: str = None) -> None:

        if njobs is None and self.njobs is not None:
            njobs = self.njobs

        dcmio.digest_dicoms(self.db_name, dicom_dir, filter_by=filter_by, njobs=njobs, log=self._logger,
                            progress_bar=progress_bar, copy_dir=copy_dir, only_digest_new=only_digest_new,
                            institution=institution)

    def convert_to_df(self, columns: list = '*', table: str = 'dicoms') -> pd.DataFrame:
        with contextlib.closing(dbu.make_connection(self.db_name)) as conn:
            with conn:
                if type(columns) is list:
                    fmt = ','.join(columns)
                else:
                    fmt = columns
                return pd.read_sql_query(f'SELECT {fmt} FROM {table}', conn)

    def compute_statistics(self, expansions: list = [0, 1, 5, 10, 15, 20],
                           atlases: list = ['atlas_anatomical_coarse',
                                            'atlas_anatomical_fine',
                                            'atlas_functional',
                                            'atlas_vascular'],
                           institutions: list = ['mskcc', 'ucsf', 'ucd', 'pmh'],
                           precompute: bool = True) -> None:
        """Compute the following statistics on each Nifti ROI
            - Volume of the structure
            - Voxel location of the centroid, for a list of expansions
            - Overlap with the structures in the atlas v1 and v2
            - Distance to white and grey matter junction

        Args:
            expansions (list, optional): List of expansions to apply to the roi overlap
                computation. Distance in mm. Defaults to [0, 1, 5, 10, 15, 20].
        """
        if precompute:
            # Pull all the nifti files in each atlas, pre-compute expansions and save to ./helper_data
            for atlas in atlases:
                query_base = 'SELECT NiftiFilePaths FROM volumes WHERE Institution LIKE '
                atlas_niftis = dbu.query_db(self.db_name, query_base + f'"{atlas}"')[0][0]
                statistics.pre_compute_masks(atlas_niftis, expansions, atlas)

            # Pull the white-grey matter files
            grey_white_niftis = dbu.query_db(self.db_name, query_base + '"grey_white_surface"')[0][0]
            statistics.pre_compute_gw_distance(grey_white_niftis)

        with dbu.make_connection(self.db_name) as conn:
            statistics_columns = dbu.list_columns(conn, 'statistics')

        '''
        Added, can delete
        '''
        atlas_shapes = {}
        for atlas in atlases:
            for distance in expansions:
                loaded = np.load(f'../helper_data/masks_{atlas}_{distance}mm.npz')['arr_0']
                atlas_name = f'{atlas}-{distance}'
                atlas_shapes[atlas_name] = loaded.shape
                temp = np.memmap(f'../helper_data/{atlas_name}.array', dtype='uint8', mode='w+', shape=loaded.shape)
                temp[:] = loaded

        interface_dist = np.load('../helper_data/interface_edt.npy', allow_pickle=True)
        brain_mask = np.array(np.load('../helper_data/gw_brain_mask.npy'), dtype='uint8')

        shared_distance = np.memmap('../helper_data/distance.array', dtype='float32', mode='w+', shape=interface_dist.shape)
        shared_brain = np.memmap('../helper_data/brain.array', dtype='uint8', mode='w+', shape=brain_mask.shape)
        shared_distance[:] = interface_dist
        shared_brain[:] = brain_mask

        fn = statistics.FindOverlapShared(columns=statistics_columns,
                                          atlas_shapes=atlas_shapes,
                                          interface_shape=interface_dist.shape,
                                          brain_shape=brain_mask.shape)
        names = {}
        for atlas in atlases:
            atlas_names = []
            with open(f'../helper_data/names_{atlas}.csv', 'r') as f:
                for row in f:
                    atlas_names.append(row.replace('\n', ''))
            names[atlas] = atlas_names

        rows = [{'Atlas': a,
                 'AtlasNames': names[a]} for a in atlases]

        with dbu.make_connection(self.db_name) as conn:
            dbu.add_rows(conn, 'atlases', rows)

        for institution in institutions:
            print(f'Computing Statistics for {institution}')

            query = f'SELECT * FROM volumes WHERE Institution LIKE "{institution}"'
            volume_rows = dbu.query_db(self.db_name, query, dicts=True)

            params = dict(desc='Computing Statistics', total=len(volume_rows))
            with Pool(processes=self.njobs) as P:
                results = list(tqdm(P.imap_unordered(fn.run, volume_rows), **params))

            rows = utils.unnest(results)

            with dbu.make_connection(self.db_name) as conn:
                dbu.add_rows(conn, 'statistics', rows)

        return None

    def summary(self, verbose: bool = False, complete: bool = False) -> pd.DataFrame:
        """Generates a summary dataframe of the cohort

        Args:
            verbose (bool, optional): Adds additional columns to the summary. Defaults to False.
            complete (bool, optional): Converts all of the dicom list to a pandas dataframe. Defaults to False.

        Returns:
            pd.DataFrame: A pandas dataframe describing the cohort.
        """
        query = "SELECT PatientID, Modality, COUNT(*) files"
        cols = ['files']
        if complete:
            with contextlib.closing(dbu.make_connection(self.db_name)) as conn:
                with conn:
                    return pd.read_sql_query('SELECT * FROM dicoms', conn)
        if verbose:
            query += ", COUNT(DISTINCT(StudyInstanceUID)) studies"
            query += ", COUNT(DISTINCT(SeriesInstanceUID)) series"
            query += ", GROUP_CONCAT(StudyDate) dates"
            query += ', GROUP_CONCAT(SeriesDescription) series_description'
            query += ', GROUP_CONCAT(ReferencedSOPInstanceUID) ref_sop'
            cols += ['studies', 'series', 'dates', 'series_description', 'ref_sop']
        query += " FROM dicoms GROUP BY PatientID, Modality"
        return self._to_df(query, cols)

    def tree(self, levels: int = None) -> utils.CustomNode:
        """Creates a tree of the cohort contents.

        Args:
            levels (int, optional): Number of levels to print.
                0 = Database name
                1 = PatientID
                2 = Frame Of Reference UID
                3 = Study Instance UID
                4 = Series Instance UID
                5 = Modality
                Defaults to all levels.

        Returns:
            utils.CustomNode: Anytree CustomNode object.
        """
        hierarchy = ['PatientID',
                     'FrameOfReferenceUID',
                     'StudyInstanceUID',
                     'SeriesInstanceUID',
                     'Modality']

        db_tree = utils.CustomNode(parent=None, node_type='Database',
                                   identifier=self.db_name)

        def find(tree, name):
            for child in tree.children:
                if child.identifier == name:
                    return child
            return None

        with contextlib.closing(dbu.make_connection(self.db_name, dicts=True)) as conn:
            with conn:
                c = conn.execute("SELECT * FROM dicoms")

                for row in c:
                    current = db_tree
                    for level, name in enumerate(hierarchy, 1):
                        if levels is not None and level > levels:
                            continue
                        match = find(current, row[name])
                        if match is None and name != 'Modality':
                            current = utils.CustomNode(parent=current, node_type=name,
                                                       identifier=row[name], level=level)
                        elif match is None:
                            utils.CustomNode(parent=current, node_type=name,
                                             identifier=row[name], modality=True, level=level)
                        elif match is not None and name != 'Modality':
                            current = match
                        else:
                            match.file_count += 1
        return db_tree

    @utils.log_notification(prefix='Saving')
    def save_dicoms(self, save_dir: str, overwrite: bool = False, prefixes: bool = True) -> None:
        """Saves the DICOM files to disk in a sorted directory structure

        Args:
            save_dir (str): The POSIX path to save the dicoms.
            overwrite (bool, optional): Allows overwriting the files if already present. Because most
                times the DICOMs are unchanged, it saves I/O to not re-write. Defaults to False.
            prefixes (bool, optional): Appends a prefix to the directories specifying UID group type.
                Defaults to True.
        """
        self.dicoms_dir = save_dir

        with contextlib.closing(dbu.make_connection(self.db_name, dicts=False)) as conn:
            with conn:
                count = conn.execute("SELECT COUNT(SOPInstanceUID) FROM dicoms").fetchone()[0]

        with contextlib.closing(dbu.make_connection(self.db_name, dicts=True)) as conn:
            with conn:
                c = conn.execute("SELECT * FROM dicoms")
                fs = dcmio.FileSaver(save_dir, overwrite, prefixes)
                if self.njobs == 1:
                    if self.progress_bar:
                        _ = list(tqdm(map(fs.save_dicoms, c), total=count, desc='Saving'))
                    else:
                        _ = list(map(fs.save_dicoms, c))
                else:
                    utils.apply_fn_parallelized(fs.save_dicoms, c, njobs=self.njobs,
                                                progress_bar=self.progress_bar, total=count)

    def remove_data(self, tags_values: dict, suffix: str = None) -> None:
        """Takes a dictionary of {tag: [values]} to remove from the cohort

        Args:
            tags_values (dict): Tag values to remove from the cohort.
            suffix (str, optional): A suffix to add to the SQL Query if further
                constraints are necessary. Defaults to None.
        """
        # Takes a dictionary of tags: [values] to remove
        with contextlib.closing(dbu.make_connection(self.db_name)) as conn:
            with conn:
                for tag, values in tags_values.items():
                    for value in values:
                        query = f'DELETE FROM dicoms WHERE {tag} = "{value}"'
                        if suffix is not None:
                            query += suffix
                        conn.execute(query)

    def remove_data_by_fn(self, tag: str, filter_fn: object) -> None:
        """Remove data from the database using a custom function which takes
            a pandas.DataFrame and returns a dictionary of which values to remove

        Args:
            tag (str): Specify the tag to iterate over the database and return queries to
                the filter_fn. For example, if you want one patient at a time, you would
                iterate over PatientID.
            filter_fn (object): A function which takes a pandas.DataFrame object and returns
                a dictionary of {tag: [value]} to remove. For example, the object could return:
                    {'SeriesInstanceUID': [1.1.1, 1.1.2, ..., 1.1.N]}
                From which all DICOMs within the tag group which match any of the SeriesInstanceUID
                values would be removed.
        """
        with contextlib.closing(dbu.make_connection(self.db_name)) as conn:
            with conn:
                c = conn.execute(f'SELECT DISTINCT({tag}) FROM dicoms')
                all_tags = [x[0] for x in c.fetchall()]
                for value in all_tags:
                    query = f'SELECT * FROM dicoms WHERE {tag} = "{value}"'
                    df = pd.read_sql_query(query, conn)
                    self.remove_data(filter_fn(df), suffix=f'AND {tag} = "{value}"')

    def number_of(self, column: str, table: str, unique: bool = True) -> list:
        with contextlib.closing(dbu.make_connection(self.db_name)) as conn:
            with conn:
                if unique:
                    c = conn.execute(f'SELECT COUNT(DISTINCT({column})) FROM {table}')
                else:
                    c = conn.execute(f'SELECT COUNT({column}) FROM {table}')
                return c.fetchone()[0]

    def iter_patients(self) -> dbu.GroupingIterator:
        return dbu.GroupingIterator(self.db_name, 'dicoms', 'PatientID')

    def iter_frames(self) -> dbu.GroupingIterator:
        return dbu.GroupingIterator(self.db_name, 'dicoms', 'FrameOfReferenceUID')

    def iter_studies(self) -> dbu.GroupingIterator:
        return dbu.GroupingIterator(self.db_name, 'dicoms', 'StudyInstanceUID')

    def iter_series(self) -> dbu.GroupingIterator:
        return dbu.GroupingIterator(self.db_name, 'dicoms', 'SeriesInstanceUID')

    def iter_volumes(self) -> dbu.GroupingIterator:
        return dbu.GroupingIterator(self.db_name, 'volumes', 'SeriesInstanceUID')

    @utils.log_notification(prefix='Assembly')
    @dbu.sql_wal_compensator
    def assemble_dicoms(self, save_dir: str, overwrite: bool = True,
                        prefixes: bool = True, save_transposed: bool = True,
                        corrections: dict = None, write_nifti: bool = False,
                        write_xarray: bool = True, only_new_volumes: bool = False,
                        single_frame_optimized: bool = False) -> None:
        """Assembles DICOMs into volumes and saves as compressed XArray NetCDF4 array (.nc) file.
            Default saving behavior is to not overwrite existing files in already present in save_dir
            and to append prefixes to the UIDs of the directory names. DICOM defaults to
            (col, row, slice) indexing, therefore, by default the volumes are saved as (row, col, slice)
            with save_transposed=True. The following corrections can be specified in a dictionary as
            True or False to apply:
                1. Hounsfield
                2. N4BiasField # Warning: Computationally expensive
                3. SUVBodyWeight
                4. NMCounts
                5. InterpolateDose

        Args:
            save_dir (str): Specifies the directory to save the .nc files.
            overwrite (bool, optional): Allows overwriting the files if already present. Defaults to True.
            prefixes (bool, optional): Appends a prefix to the directories specifying UID group type.
                Defaults to True.
            save_transposed (bool, optional): Saves volumes as (row, col, slice). Defaults to True.
            corrections (dict, optional): Correction factors to apply to the assembled volumes. Defaults to None.
            progress_bar (bool, optional): Displays progress bar during dicom assembly. Defaults to True.
            write_nifti (bool, optional): Writes a .nii.gz file in adition to the .nc files. Defaults to False.
            write_xarray (bool, optional): Writes a .nc file. Needed for augmentations. Defaults to True.
            only_new_volumes (bool, optional): Assembles only new volumes. Defaults to False.
            single_frame_optimized (bool, optional): Optimized reconstruction for all data entirely within a
                signle frame of reference. Defaults to False.
        """
        am = assembly.Assembler(save_dir=save_dir, db_name=self.db_name,
                                overwrite=overwrite, prefixes=prefixes,
                                save_transposed=save_transposed, log=self._logger,
                                corrections=corrections, write_nifti=write_nifti,
                                write_xarray=write_xarray, only_new_volumes=only_new_volumes,
                                single_frame_optimized=single_frame_optimized,
                                njobs=self.njobs, progress_bar=self.progress_bar)

        n = self.number_of('FrameOfReferenceUID', 'dicoms')
        if self.njobs == 1 or single_frame_optimized:
            if self.progress_bar and not single_frame_optimized:
                return list(tqdm(map(am.assemble, self.iter_frames()), desc='Assembly', total=n))
            return list(map(am.assemble, self.iter_frames()))
        else:
            with Pool(self.njobs) as P:
                if self.progress_bar:
                    return list(tqdm(P.imap_unordered(am.assemble, self.iter_frames()), desc='Assembly', total=n))
                return list(P.imap_unordered(am.assemble, self.iter_frames()))

    @utils.log_notification(prefix='Applying Tools')
    @dbu.sql_wal_compensator
    def apply_tools(self, tools):
        th = augmentations.ToolHandler(tools=tools, db_name=self.db_name, log=self._logger)
        if self.njobs == 1:
            return list(map(th.apply, self.iter_volumes()))
        with Pool(self.njobs) as P:
            print('Apply tools')
            if self.progress_bar:
                n = self.number_of('SeriesInstanceUID', 'volumes')
                return list(tqdm(P.imap_unordered(th.apply, self.iter_volumes()), desc='Tools', total=n))
            return list(P.imap_unordered(th.apply, self.iter_volumes()))

    @utils.save_dir_check
    def new_rtstruct_from_image(self, image_series_uid: str, masks: list,
                                roi_names: list = None, *,
                                save_dir: str = None, overwrite: list = False,
                                prefixes: bool = True):
        kwargs = dict(image_series_uid=image_series_uid, masks=masks,
                      roi_names=roi_names, save_dir=save_dir,
                      overwrite=overwrite, prefixes=prefixes)
        self.rtstruct_creator._apply_fn_and_write(self.rtstruct_creator.new_from_image, **kwargs)

    @utils.save_dir_check
    def new_rtstruct_from_rtstruct(self, rtstruct: str, image_series_uid: str,
                                   masks: list, roi_names: list = None, *,
                                   save_dir: str = None, overwrite: bool = False,
                                   prefixes: bool = True):
        kwargs = dict(rtstruct=rtstruct, image_series_uid=image_series_uid,
                      masks=masks, roi_names=roi_names, save_dir=save_dir,
                      overwrite=overwrite, prefixes=prefixes)
        self.rtstruct_creator._apply_fn_and_write(self.rtstruct_creator.new_from_rtstruct, **kwargs)

    @utils.save_dir_check
    def append_to_rtstruct(self, rtstruct: str, image_series_uid: str, masks: list,
                           roi_names: list = None, *,
                           save_dir: str = None, overwrite: bool = False,
                           prefixes: bool =True):
        kwargs = dict(rtstruct=rtstruct, image_series_uid=image_series_uid,
                      masks=masks, roi_names=roi_names, save_dir=save_dir,
                      overwrite=overwrite, prefixes=prefixes)
        self.rtstruct_creator._apply_fn_and_write(self.rtstruct_creator.append_to_rtstruct, **kwargs)


def load_database(db_name: str, **kwargs) -> Cohort:
    """Loads a database from disk and returns the functional cohort object

    Args:
        db_name (str): A POSIX path to a generated database
        njobs (int, optional): Number of parallel jobs to allow the cohort object ot run. Defaults to the
            number of logical system cores.

    If any non-default changes were made, the following may be required:
        config_file (str, optional): A configuration of the database tables. Defaults to 'table_config.yaml'.
        filter_by (str, optional): A yaml file specifying which DICOMs to digest into the
            database. Defaults to 'filter_by.yaml.

    Returns:
        Cohort: The loaded database in the form of a Cohort object
    """
    kwargs['dicom_dir'] = None
    return Cohort(db_name=db_name, **kwargs)
