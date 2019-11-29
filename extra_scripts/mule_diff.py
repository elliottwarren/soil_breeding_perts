"""
Difference of all common fields within two UM files
"""

import sys
import mule
from mule.operators import SubtractFieldsOperator

subtract_operator = SubtractFieldsOperator()

# filenames passed in from shell environment
# input filenames
InputFile1 = sys.argv[1]
InputFile2 = sys.argv[2]
# output filename
OutputFile = sys.argv[3]

# ff1 = mule.FieldsFile.from_file("InputFile1")
# ff2 = mule.FieldsFile.from_file("InputFile2")

ff1 = mule.load_umfile(InputFile1)
ff1 = mule.load_umfile(InputFile2)

ff_out = ff1.copy()

for field_1 in ff1.fields:
    if field_1.lbrel in (2,3) and field_1.lbuser4 != 30:
        for field_2 in list(ff2.fields):
            if field_2.lbrel not in (2,3):
                ff2.fields.remove(field_2)
                continue
            elif ((field_1.lbuser4 == field_2.lbuser4) and
                  (field_1.lbft == field_2.lbft) and
                  (field_1.lblev == field_2.lblev)):
                ff_out.fields.append(subtract_operator([field_1, field_2]))
                ff2.fields.remove(field_2)
                break
        else:
            ff_out.fields.append(field_1)
    else:
        ff_out.fields.append(field_1)

ff_out.to_file("OutputFile")