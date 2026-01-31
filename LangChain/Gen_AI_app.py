{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7a6dac23",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Your_API_KEY\n",
      "Generative-AI application\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "load_dotenv(dotenv_path=\".env\", override=True)\n",
    "\n",
    "os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')\n",
    "os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')\n",
    "os.environ['LANGSMITH_TRACING'] ='true'\n",
    "os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')\n",
    "\n",
    "print(os.environ['LANGSMITH_API_KEY'])\n",
    "print(os.environ['LANGSMITH_PROJECT'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8cfd334e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9371c3bf",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d89cac0",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "756f1941",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d6ef9418",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "myenv",
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
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
