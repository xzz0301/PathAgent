"""Provision paper-compatible caption scorers and a user-local Java runtime."""
import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--server',type=Path,required=True);a=p.parse_args()
    root=a.server/'tools';root.mkdir(exist_ok=True)
    subprocess.run([sys.executable,'-m','pip','install','pycocoevalcap==1.2','nltk==3.9.2'],check=True)
    java=root/'jre/bin/java'
    if not java.exists():
        url='https://api.azul.com/metadata/v1/zulu/packages/?java_version=17&os=linux&arch=x86_64&archive_type=tar.gz&java_package_type=jre&latest=true&release_status=ga&availability_types=CA'
        packages=json.load(urllib.request.urlopen(url,timeout=30))
        package=next(x for x in packages if '-fx-' not in x['name'] and 'musl' not in x['name'])
        archive=root/package['name'];temporary=archive.with_suffix('.downloading')
        with urllib.request.urlopen(package['download_url'],timeout=60) as response,temporary.open('wb') as target:
            while data:=response.read(1024*1024):target.write(data)
        temporary.replace(archive)
        with tarfile.open(archive) as source:
            base=root.resolve()
            for member in source.getmembers():
                dest=(root/member.name).resolve()
                if not dest.is_relative_to(base):raise ValueError('Unsafe archive path')
                if member.issym() and not (dest.parent/member.linkname).resolve().is_relative_to(base):
                    raise ValueError('Unsafe archive link')
                if member.islnk() and not (root/member.linkname).resolve().is_relative_to(base):
                    raise ValueError('Unsafe archive hardlink')
            top=source.getmembers()[0].name.split('/')[0]
            source.extractall(root)
        (root/'jre').symlink_to(root/top,target_is_directory=True)
        (root/'jre_download.json').write_text(json.dumps({'source':package,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))
    subprocess.run([str(java),'-version'],check=True)
    import os
    os.environ['PATH']=str(java.parent)+':'+os.environ['PATH']
    from pycocoevalcap.meteor.meteor import Meteor
    scorer=Meteor()
    try:
        score,_=scorer.compute_score({'test':['normal tissue']},{'test':['normal tissue']})
        if not score>0:raise ValueError('METEOR smoke score invalid')
    finally:
        if scorer.meteor_p.poll() is None:scorer.meteor_p.kill();scorer.meteor_p.wait()
    (root/'wsi_metrics_ready.json').write_text(json.dumps({'meteor_smoke_score':score,'pycocoevalcap':'1.2','nltk':'3.9.2','java':str(java)},indent=2))


if __name__=='__main__':main()
