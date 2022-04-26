import openpifpaf

from . import power_drill_kp


def register():
    openpifpaf.DATAMODULES['power_drill'] = power_drill_kp.PowerDrillKp
