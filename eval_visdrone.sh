DetJSON=$1

python visdrone/json_to_txt.py --out .visdrone_det_txt --gt-json data/visdrone/coco_format/annotations/val_label.json --det-json $DetJSON
python visdrone_eval/evaluate.py --dataset-dir data/visdrone/VisDrone2019-DET-val --res-dir .visdrone_det_txt
rm -rf .visdrone_det_txt