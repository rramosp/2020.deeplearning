course_id = '2020.deeplearning'
github_repo = 'rramosp/%s'%course_id
zip_file_url="https://github.com/%s/archive/master.zip"%github_repo

def get_github_last_commit_date():
    try:
        import requests
        import dateutil.parser
        from datetime import timezone

        r = requests.get('https://api.github.com/repos/%s/commits/master'%github_repo).json()
        date = dateutil.parser.parse(r["commit"]["committer"]["date"]).astimezone(timezone.utc)
        return date
    except Exception as e:
        return None

def get_last_modif_date(localdir):
    try:
        import time, os, pytz
        import datetime
        k = datetime.datetime.fromtimestamp(max(os.path.getmtime(root) for root,_,_ in os.walk(localdir)))
        localtz = datetime.datetime.now(datetime.timezone(datetime.timedelta(0))).astimezone().tzinfo
        k = k.astimezone(localtz)
        return k
    except Exception as e:
        return None
    
import requests, zipfile, io, os, shutil
def init(force_download=False):
    ghdate = get_github_last_commit_date()
    localdate = get_last_modif_date('.')    
    if force_download or ghdate is None or localdate is None or ghdate>localdate:
        print ("replacing local resources")
        dirname = course_id+"-master/"
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        if os.path.exists("local"):
            shutil.rmtree("local")
        shutil.move(dirname+"/local", "local")
        shutil.rmtree(dirname)
