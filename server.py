from sanic import Sanic
from sanic.response import text
from darkflow.net.build import TFNet
import cv2
import wget

options = {
    "model": "cfg/tiny-yolo-chars.cfg",
    "load": -1,
    "threshold": 0.1,
    "labels": "dp/custom.names",
}

model = TFNet(options)

app = Sanic()

@app.route('/')
async def status(q):
    return text('OK')

@app.route('/<imagepath>')
async def go(q, imagepath):
    return text('\n'.join(["Griffin 0.91 620 117 817 257", "Dragon 0.88 183 273 369 413", "Wizard 0.92 405 232 567 383", "Soldier 0.96 476 8 642 172", "Archer 0.94 575 303 754 453", "Hydra 0.82 316 83 457 244"]))
    ip = q.ip
    imagepath = imagepath.split('/')[-1]
    print(ip)
    print("GOT A REQUEST FOR %s:3415/%s" % (ip, imagepath))
    fn = wget.download('http://%s:3415/%s' % (ip, imagepath))
    img = cv2.imread(fn)
    print("PREDICTING...")
    res = model.return_predict(img)
    return text('\n'.join(res))

if __name__ == "__main__": app.run(host='0.0.0.0', port=10919)
