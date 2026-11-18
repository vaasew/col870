import argparse
import os
import time
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('--port',type=int,required=True)
p.add_argument('--pid',type=int,required=True)
p.add_argument('--timeout',type=float,default=1800)
a=p.parse_args()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
end=time.monotonic()+a.timeout
while time.monotonic()<end:
    try:
        os.kill(a.pid,0)
    except ProcessLookupError:
        raise SystemExit('Model server exited before becoming ready. Inspect server.log.')
    try:
        with opener.open(f'http://127.0.0.1:{a.port}/health',timeout=2) as r:
            if r.status==200:
                print('Model server ready',flush=True)
                break
    except OSError:
        time.sleep(2)
else:
    raise TimeoutError('Model startup timed out; inspect server.log')
