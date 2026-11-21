"""
URL configuration of the replay viewer.

The authors' routes for live simulation (simulator_home, process_environment,
update_environment, path_tester) are not used: the backend here is headless
and runs on its own. The replay routes are kept as they were; `url()` (removed
in Django 4) became `re_path()`.
"""
from django.urls import re_path

from translator import views as translator_views

urlpatterns = [
  re_path(r'^$', translator_views.landing, name='landing'),
  re_path(r'^demo/(?P<sim_code>[\w-]+)/(?P<step>[\w-]+)/(?P<play_speed>[\w-]+)/$',
          translator_views.demo, name='demo'),
  re_path(r'^replay_persona_state/(?P<sim_code>[\w-]+)/(?P<step>[\w-]+)/(?P<persona_name>[\w-]+)/$',
          translator_views.replay_persona_state, name='replay_persona_state'),
]
