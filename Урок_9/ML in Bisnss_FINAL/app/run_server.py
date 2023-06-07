{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "68fc6745-8e23-43c9-9c79-22dd9954ad1d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Record_ID' 'Auction_ID' 'Bidder_ID' 'Bidder_Tendency' 'Bidding_Ratio'\n",
      " 'Successive_Outbidding' 'Last_Bidding' 'Auction_Bids'\n",
      " 'Starting_Price_Average' 'Early_Bidding' 'Winning_Ratio'\n",
      " 'Auction_Duration']\n",
      "* Loading the model and Flask starting server...please wait until server has fully started\n",
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on all addresses (0.0.0.0)\n",
      " * Running on http://127.0.0.1:8181\n",
      " * Running on http://192.168.1.217:8181\n",
      "Press CTRL+C to quit\n",
      " * Restarting with stat\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    }
   ],
   "source": [
    "#%tb\n",
    "# USAGE\n",
    "# Start the server:\n",
    "# \tpython run_server.py\n",
    "# Submit a request via Python:\n",
    "#\tpython simple_request.py\n",
    "\n",
    "# import the necessary packages\n",
    "import pickle\n",
    "import pandas as pd\n",
    "import os\n",
    "#import cloudpickle\n",
    "import flask\n",
    "import logging\n",
    "from logging.handlers import RotatingFileHandler\n",
    "import time\n",
    "\n",
    "from model_transforms import NumberTaker, ExperienceTransformer, NumpyToDataFrame\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "\n",
    "# initialize our Flask application and the model\n",
    "app = flask.Flask(__name__)\n",
    "model = None\n",
    "\n",
    "handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)\n",
    "logger = logging.getLogger(__name__)\n",
    "logger.setLevel(logging.INFO)\n",
    "logger.addHandler(handler)\n",
    "\n",
    "def load_model(model_path):\n",
    "\t# load the pre-trained model\n",
    "\t# global model\n",
    "\twith open(model_path, 'rb') as f:\n",
    "\t\tmodel = pickle.load(f)\n",
    "\tprint(model)\n",
    "\treturn model\n",
    "\n",
    "modelpath = \"C:\\\\Users\\\\SAMOL\\\\000 Машинное обучение в бизнесе\\Урок_9\\\\ML in Bisnss_FINAL\\\\app\\\\models\\\\ctb_clf.pkl\"\n",
    "load_model(modelpath)\n",
    "\n",
    "\n",
    "@app.route(\"/\", methods=[\"GET\"])\n",
    "def general():\n",
    "\treturn \"\"\"Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST\"\"\"\n",
    "\n",
    "@app.route(\"/predict\", methods=[\"POST\"])\n",
    "def predict():\n",
    "\t# initialize the data dictionary that will be returned from the\n",
    "\t# view\n",
    "\tresponse = {'success': False}\n",
    "\tcurr_time = time.strftime('[%Y-%b-%d %H:%M:%S]')\n",
    "\n",
    "\t# ensure an image was properly uploaded to our endpoint\n",
    "\tif flask.request.method == 'POST':\n",
    "\t\trequest_json = flask.request.get_json()\n",
    "\n",
    "        \n",
    "\t\tinput_data = pd.DataFrame({\n",
    "\t\t\t'auctionid': [request_json.get('auctionid', '')],\n",
    "\t\t\t'bidderid': [request_json.get('bidderid', '')],\n",
    "\t\t\t'bid': [(request_json.get('bid', ''))],\n",
    "\t\t\t'bidderrate': [(request_json.get('bidderrate', ''))],\n",
    "\t\t\t'openbid': [(request_json.get('openbid', ''))],\n",
    "\t\t\t'price': [(request_json.get('price', ''))],\n",
    "\t\t\t'daystolive': [(request_json.get('daystolive', ''))],\n",
    "\t\t\t'hourstolive': [(request_json.get('hourstolive', ''))],\n",
    "\t\t\t'finalprice': [(request_json.get('finalprice', ''))],\n",
    "\t\t\t'itemtype': [request_json.get('itemtype', '')]\n",
    "\t\t}, index=[0])\n",
    "\t\tlogger.info(f'{curr_time} Data: {input_data.to_dict()}')\n",
    "\n",
    "\t\ttry:\n",
    "\t\t\t# Predict the result\n",
    "\t\t\tpreds = model.predict_proba(input_data)\n",
    "\t\t\tresponse['predictions'] = round(preds[:, 1][0], 5)\n",
    "\t\t\t# Request successful\n",
    "\t\t\tresponse['success'] = True\n",
    "\t\texcept AttributeError as e:\n",
    "\t\t\tlogger.warning(f'{curr_time} Exception: {str(e)}')\n",
    "\t\t\tresponse['predictions'] = str(e)\n",
    "\t\t\t# Request unsuccessful\n",
    "\t\t\tresponse['success'] = False\n",
    "\n",
    "\t# вернуть словарь данных в виде ответа JSON\n",
    "\treturn flask.jsonify(response)\n",
    "\n",
    "# если это основной поток выполнения, сначала загрузите модель и\n",
    "# затем запускаем сервер\n",
    "if __name__ == \"__main__\":\n",
    "\tprint((\"* Loading the model and Flask starting server...\"\n",
    "\t\t\"please wait until server has fully started\"))\n",
    "\tport = int(os.environ.get('PORT', 8181))\n",
    "\tapp.run(host='0.0.0.0', debug=True, port=port)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4d5942dc-2bf5-4ca7-b204-c19d78d465b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "#%tb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8cf84489-9410-4a50-b6b5-6700ac75aea0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
