{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a63986e2-6f0b-4b69-a1bb-da85048c350d",
   "metadata": {},
   "source": [
    "В файле model_pipeline.py (или аналогичном файле) находится код, отвечающий за обучение модели и сохранение модели в файл ctb_clf.pkl. Обычно это включает в себя следующие шаги:\n",
    "\n",
    "- Загрузка данных, подготовка признаков и целевой переменной.\n",
    "- Определение и настройка модели.\n",
    "- Обучение модели на обучающих данных.\n",
    "- Оценка качества модели на тестовых данных (опционально).\n",
    "- Сохранение обученной модели в файл ctb_clf.pkl.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9f656f0d-4e88-42d5-9e41-3b1cba222f3e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9936758893280633\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['ctb_clf.pkl']"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.feature_selection import SelectKBest, f_classif\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from io import StringIO\n",
    "import requests\n",
    "import joblib\n",
    "\n",
    "# Загрузка данных\n",
    "url = \"https://archive.ics.uci.edu/ml/machine-learning-databases/00562/Shill Bidding Dataset.csv\"\n",
    "data_text = requests.get(url).text\n",
    "data_file = StringIO(data_text)\n",
    "data = pd.read_csv(data_file)\n",
    "\n",
    "# Разделение на признаки и целевую переменную\n",
    "X = data.drop(\"Class\", axis=1)  # Признаки\n",
    "y = data[\"Class\"]  # Целевая переменная\n",
    "\n",
    "# Разделение на обучающую и тестовую выборки\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)\n",
    "\n",
    "# Определение этапов предобработки данных и выбора признаков\n",
    "preprocessor = ColumnTransformer([\n",
    "    ('numeric', StandardScaler(), ['Auction_Duration','Bidder_Tendency','Bidding_Ratio','Successive_Outbidding',\n",
    "                   'Last_Bidding','Auction_Bids','Starting_Price_Average','Early_Bidding','Winning_Ratio']),  \n",
    "    # Добавьте другие этапы предобработки данных, если необходимо\n",
    "])\n",
    "\n",
    "# Определение классификатора\n",
    "classifier = RandomForestClassifier()\n",
    "\n",
    "# Создание пайплайна\n",
    "model_pipeline = Pipeline([\n",
    "    ('preprocessor', preprocessor),\n",
    "    ('feature_selector', SelectKBest(score_func=f_classif, k='all')),  # Используем k='all' для всех признаков\n",
    "    ('classifier', classifier)\n",
    "])\n",
    "\n",
    "# Обучение модели\n",
    "model_pipeline.fit(X_train, y_train)\n",
    "\n",
    "# Оценка качества модели на тестовой выборке\n",
    "accuracy = model_pipeline.score(X_test, y_test)\n",
    "print(\"Accuracy:\", accuracy)\n",
    "\n",
    "# Сохранение модели в файл\n",
    "joblib.dump(model_pipeline, 'ctb_clf.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "16296b2a-d3da-4abe-bff4-370bcdc17b66",
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
