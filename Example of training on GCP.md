#Example of training on GCP 

```

gcloud ml-engine models list
```

to show the exist models on GCP. 

```
TRAIN_DATA=$(pwd)/data/adult.data.csv
EVAL_DATA=$(pwd)/dara/adult.test.csv
```

set the train data and evlauation data to local file pats.

```
$(pwd) //print the current director path. 
```

running following command to run ml-engine for local trainer.

```
gcloud ml-engine local train --module-name trainer.task --package-path trainer --job-dir $MODEL_DIR -- --train-files $TRAIN_DATA --eval-files $EVAL_DATA --train-steps 1000 --eval-steps 100
```

lauch tensorboard for visual board of trained model.

Open tensorboard on cloud shell. Then click "web viewer" select "open web by port 8080"

```
tensorboard --logdir=$MODEL_DIR --port=8080
```

Open tensorfboard on local command line.

```
tensorboard --logdir=$MODEL_DIR
```

