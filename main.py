import json
import constants as c
import utils.common as common
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.observers import TelegramObserver

ex = Experiment()

with open('mongodb.json') as f:
    mongodb_settings = json.load(f)

ex.observers.append(
    MongoObserver.create(
        url='mongodb://{user}:{pwd}@{ip}:{port}'.format(**mongodb_settings),
        db_name='{db}'.format(**mongodb_settings),
    )
)

if (c.ROOT / 'telegram.json').exists():
    ex.observers.append(
        TelegramObserver.from_config('telegram.json')
    )

for filename in c.ROOT.glob('**/*.py'):
    print("Saving File: {}".format(filename.absolute()))
    ex.add_source_file(filename.absolute())

@ex.config
def default_config():
    project = 'openmic'

@ex.config
def custom_config(project):
    ex.add_config(str(c.ROOT / 'configs' / '{}.json'.format(project)))

@ex.automain
def run( _config, _run, _rnd):
    # experiment, data_set, model, training, resources, _run, _rnd
    experiment = common.load_class(
        _config['class_'],
        ex,
        _run,
        _rnd,
        **_config
    )

    return experiment.run()
