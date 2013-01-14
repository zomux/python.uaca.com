/root/apps/uwsgi-1.3/uwsgi \
--chdir=/home/python.uaca.com \
--module=uaca.wsgi:application \
--env DJANGO_SETTINGS_MODULE=uaca.settings \
--master --pidfile=/tmp/uwsgi.pid \
--socket=/tmp/uwsgi.sock \
--processes=2 \
--uid=1000 --gid=2000 \
--harakiri=20 \
--limit-as=128 \
--max-requests=5000 \
--vacuum \
--home=/home/python.uaca.com/virtualenv \
--daemonize=/var/log/uwsgi.log

# --socket=/tmp/uwsgi.sock       # can also be a file 
# --processes=2                  # number of worker processes 
# --uid=1000 --gid=2000          # if root, uwsgi can drop privileges 
# --harakiri=20                  # respawn processes taking more than 20 seconds 
# --limit-as=128                 # limit the project to 128 MB 
# --max-requests=5000            # respawn processes after serving 5000 requests 
# --vacuum                       # clear environment on exit 
# --home=/home/python.uaca.com/virtualenv    # optional path to a virtualenv 
# --daemonize=/var/log/uwsgi.log      # background the process