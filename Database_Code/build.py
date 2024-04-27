from dicomanager.cohort import Cohort
from dicomanager.cohort import load_database
from datetime import datetime
from dicomanager import utils
import warnings
warnings.simplefilter("ignore", UserWarning)


if __name__ == '__main__':
    # Build cohort
    # Cohort buliding is multithreaded, but on mac the SQL database thread locking does not always behave
    # If it hangs, specify njobs=1 to revert to single-threaded tasks
    PROJECT_DIR = '/project_directory_path/'
    ISODATE = datetime.now().isoformat()

    MAIN = f'{PROJECT_DIR}/sql_db/Main_{ISODATE}.db'
    LOGFILE = f'{PROJECT_DIR}/sql_db/AssemblyNotifications_{ISODATE}.log'

    NJOBS = 48
    INSTITUTION = 'InstitutionName'
    BUILD_NEW_DB = False

    cohort = None
    with utils.Timer(prefix='Constructing All'):
        dicom_dir = f'{PROJECT_DIR}/unsorted/{INSTITUTION}'
        copy_dir = f'{PROJECT_DIR}/dicoms/'
        volume_dir = f'{PROJECT_DIR}/volumes/'

        if BUILD_NEW_DB:
            cohort = Cohort(dicom_dir=dicom_dir,
                            db_name=MAIN,
                            copy_dir=copy_dir,
                            njobs=NJOBS,
                            institution=INSTITUTION,
                            logfile=LOGFILE)
        else:
            cohort = load_database(MAIN)
            cohort.add_dicoms(dicom_dir=dicom_dir,
                              copy_dir=copy_dir,
                              institution=INSTITUTION)

        cohort.assemble_dicoms(save_dir=volume_dir,
                               write_nifti=True,
                               write_xarray=True,
                               only_new_volumes=True,
                               single_frame_optimized=True)

    with utils.Timer(prefix='Computing Statistics'):
        cohort.compute_statistics(atlases=[])