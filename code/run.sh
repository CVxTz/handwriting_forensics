#register in http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database
#Download data/lineImages-all.tar.gz
#decompress it in ../input/

python clean_images.py
python parse_forms.py
python handwriting_resnet.py