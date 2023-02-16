import os
import tqdm
import json


import argparse


class Json2Txt(object):

    def __init__(self, gt_json, det_json, out_dir):
        gt_data = json.load(open(gt_json))
        self.images = {x['id']: {'file': x['file_name'], 'height':x['height'], 'width':x['width']} for x in gt_data['images']}

        det_data = json.load(open(det_json))
        
        self.results = {}
        for result in det_data:
            if result['image_id'] not in self.results.keys():
                self.results[result['image_id']] = []
            self.results[result['image_id']].append({'box': result['bbox'], 'category': result['category_id'], 'score': result['score']})
        
        self.out_dir = out_dir
    
    def to_txt(self):
        for img_id in tqdm.tqdm(self.images.keys()):
            file_name = self.images[img_id]['file'].replace('jpg', 'txt')
            with open(os.path.join(self.out_dir, file_name), 'w') as fw:
                for pred in self.results[img_id]:
                    row = '%.2f,%.2f,%.2f,%.2f,%.8f,%d,-1,-1'%(pred['box'][0],pred['box'][1],pred['box'][2],pred['box'][3],pred['score'],pred['category'])
                    fw.write(row+'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--out', required=True, type=str, help='output txt dir')
    parser.add_argument('--gt-json', required=False, type=str, default='visdrone_data/annotations/val_label', help='Grond Truth Info JSON')
    parser.add_argument('--det-json', required=True, type=str, help='COCO style result JSON')
    args = parser.parse_args()

    gt_json = args.gt_json
    det_json = args.det_json
    outdir = args.out
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        
    print('Json to txt:', outdir)
    tool = Json2Txt(gt_json, det_json, outdir)
    tool.to_txt()
    










































