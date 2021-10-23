import os
import pandas as pd

# Press the green button in the gutter to run the script.
working_dir = r"F:\Learning\Schneider\ml-assignment-main\Data Scientist _SE_assignment\test"
directory_contents = os.listdir(working_dir)

df_email = pd.DataFrame(columns=['classname', 'email_msg'])
for eachfolder in directory_contents:
    mypath = os.path.join(working_dir, eachfolder)
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

    for i in range(len(onlyfiles)):
        readfile = open(os.path.join(mypath, onlyfiles[i]))

        alllines = "".join(readfile.readlines())

        readfile.close()

        dict1 = {'classname': eachfolder, 'email_msg': alllines}

        df_email = df_email.append(dict1, ignore_index=True)

print(len(df_email))
df_email.to_csv('../data/test.csv', index=False)