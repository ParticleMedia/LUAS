## Prepare Training data for Llama SFT

### WOZ2.2

download data from: `https://github.com/budzianowski/multiwoz.git`

run the following command
```bash
python reformat_woz22_00.py
python reformat_woz22_01.py

# new data will be generated at ./woz.2.2.real
```


### WOZ2.4

download data from: `https://github.com/smartyfh/MultiWOZ2.4.git`

> Note: woz2.4 need more normalization based on the README.md from git `MultiWOZ2.4.git`

run the following command
```bash
# cd MultiWOZ2.4.git/data
# unzip MULTIWOZ2.4.zip 
# unzip dev_test_refined.zip
# cd ../
# python create_data.py

python reformat_woz24.py

# new data will be generated at ./woz.2.4.real
```


### Generated data

the WOZ2.2 and WOZ2.4 share the same version of generation data except the case for the characters.

run 
```bash
python run convert_gen_to_sft.py --woz22 --woz24

# new data will be generated at './woz.2.2.gen' and './woz.2.4.gen'
```