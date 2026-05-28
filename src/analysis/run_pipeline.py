from analysis.pipeline_config import PipelineConfig
from analysis.pipeline import HaloPipeline
import sys
config_file = sys.argv[1]

config = PipelineConfig.from_yaml(config_file)
pipe   = HaloPipeline(config)
pipe.run_all()
