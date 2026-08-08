@echo off
set MLFLOW_ALLOW_FILE_STORE=true
echo MLflow baslatiliyor... Lutfen tarayicinizdan http://127.0.0.1:5001 adresine gidin.
echo Bu pencereyi kapatirsaniz MLflow kapanir!
mlflow ui --port 5001
pause
