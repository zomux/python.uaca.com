from django.conf.urls import patterns, include, url
from django.conf import settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
    (r'^$', 'main.views.index'),
    (r'^ahpp/$', 'ahpp.views.index'),
    # Examples:
    # url(r'^$', 'uaca.views.home', name='home'),
    # url(r'^uaca/', include('uaca.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # url(r'^admin/', include(admin.site.urls)),
     (r'^static/(?P<path>.*)$','django.views.static.serve', dict(document_root=settings.STATIC_ROOT)),
)
