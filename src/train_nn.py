import mlflow
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
np.random.seed(42)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data import runDataProcessing, load_processed_data
from evaluate import classification_metrics, regression_metrics
from utils import log_confusion_matrix, log_residual_plot, random_search_trial_c, random_search_trial_r, log_classification_learning_curve, log_regression_learning_curve, log_nn_ablation, log_feature_importance

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Student Performance NN Models")

def train_nn_classifier(X_train, y_train, X_val, y_val, X_test, y_test, best_params, name="Final_KerasNN_Classifier"):
    if mlflow.active_run() is not None:
        mlflow.end_run()
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    for size in best_params["hidden_layer_sizes"]:
        if best_params["activation"] == "leaky_relu":
            model.add(layers.Dense(size, kernel_regularizer=regularizers.l2(best_params["l2"])))
            model.add(layers.LeakyReLU(negative_slope=0.1))
        else:
            model.add(layers.Dense(size, activation=best_params["activation"],
                                   kernel_regularizer=regularizers.l2(best_params["l2"])))
        model.add(layers.Dropout(best_params["dropout"]))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_params["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    if X_val is None:
        history = model.fit(
            X_train, y_train,
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            verbose=0
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=best_params["epochs"],
            validation_data=(X_val, y_val),
            batch_size=best_params["batch_size"],
            verbose=0
        )

    log_classification_learning_curve(history, name)

    preds = (model.predict(X_test) > 0.5).astype(int).flatten()
    metrics = classification_metrics(y_test, preds)

    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        log_confusion_matrix(y_test, preds, name)
        mlflow.keras.log_model(
            model,
            artifact_path=name,
            registered_model_name=name
        )
    

    return model

def train_nn_regression(X_train, y_train, X_val, y_val, X_test, y_test, best_params, name="Final_KerasNN_Regression"):
    if mlflow.active_run() is not None:
        mlflow.end_run()

    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    for size in best_params["hidden_layer_sizes"]:
        if best_params["activation"] == "leaky_relu":
            model.add(layers.Dense(size, kernel_regularizer=regularizers.l2(best_params["l2"])))
            model.add(layers.LeakyReLU(negative_slope=0.1))
        else:
            model.add(layers.Dense(size, activation=best_params["activation"],
                                   kernel_regularizer=regularizers.l2(best_params["l2"])))
        model.add(layers.Dropout(best_params["dropout"]))

    model.add(layers.Dense(1))  # regression output

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_params["lr"]),
        loss="mse",
        metrics=["mae"]
    )

    if X_val is None:
        history = model.fit(
            X_train, y_train,
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            verbose=0
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=best_params["epochs"],
            validation_data=(X_val, y_val),
            batch_size=best_params["batch_size"],
            verbose=0
        )

    log_regression_learning_curve(history, name)

    preds = model.predict(X_test).flatten()
    metrics = regression_metrics(y_test, preds)

    baseline_mae = mean_absolute_error(y_test, preds)
    log_nn_ablation(model, X_test, y_test, name, baseline_mae)


    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        log_residual_plot(y_test, preds, name)
        mlflow.keras.log_model(
            model,
            artifact_path=name,
            registered_model_name=name
        )

    

    return model


def main():

    # 1. Process data
    runDataProcessing()

    # 2. Load classification data
    train_class = load_processed_data("data/processed/splits/train_classification.csv")
    val_class = load_processed_data("data/processed/splits/val_classification.csv")
    test_class = load_processed_data("data/processed/splits/test_classification.csv")

    # 3. Load regression data
    train_regres = load_processed_data("data/processed/splits/train_regression.csv")
    val_regres = load_processed_data("data/processed/splits/val_regression.csv")
    test_regres = load_processed_data("data/processed/splits/test_regression.csv")

    # 4. Build classification splits
    global X_train_c, y_train_c, X_val_c, y_val_c, X_test_c, y_test_c
    X_train_c = train_class.drop(columns=["G3", "Pass"]).values
    y_train_c = train_class["Pass"].values

    X_val_c = val_class.drop(columns=["G3", "Pass"]).values
    y_val_c = val_class["Pass"].values

    X_test_c = test_class.drop(columns=["G3", "Pass"]).values
    y_test_c = test_class["Pass"].values

    # 5. Build regression splits
    global X_train_r, y_train_r, X_val_r, y_val_r, X_test_r, y_test_r
    X_train_r = train_regres.drop(columns=["G3", "Pass"]).values
    y_train_r = train_regres["G3"].values

    X_val_r = val_regres.drop(columns=["G3", "Pass"]).values
    y_val_r = val_regres["G3"].values

    X_test_r = test_regres.drop(columns=["G3", "Pass"]).values
    y_test_r = test_regres["G3"].values


    param_space = {
        "learning_rate": [0.0001, 0.001, 0.005, 0.01],
        "batch_size": [16, 32, 64],
        "hidden_layer_sizes": [
                    (16, 4),
                    (16, 8),
                    (8, 4),
                    (8, 16),
                ],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "l2": [0.0, 0.0001, 0.001, 0.01],
        "activation": ["relu", "tanh", "leaky_relu"],
        "epochs": [100, 125, 150] 
    }

    best_c = None
    for _ in range(5):
        result = random_search_trial_c(X_train_c, y_train_c, param_space, k=3)
        print(result)
        if best_c is None or result["avg_f1"] > best_c["avg_f1"]:
            best_c = result

    print("Best hyperparameters:", best_c)

    # Check it on the validation split
    train_nn_classifier(X_train_c, y_train_c, X_val_c, y_val_c, X_val_c, y_val_c, best_c["params"], name="Final_KerasNN_Classifier_validation")


    # # Merge train + validation
    X_final_c = np.concatenate([X_train_c, X_val_c], axis=0)
    y_final_c = np.concatenate([y_train_c, y_val_c], axis=0)

    # # # Train final NN model on test split with validation data
    train_nn_classifier(X_final_c, y_final_c, None, None, X_test_c, y_test_c, best_c["params"])

    best_r = None
    for _ in range(5):
        result = random_search_trial_r(X_train_r, y_train_r, param_space, k=3)
        print(result)
        if best_r is None or result["avg_mae"] < best_r["avg_mae"]:
            best_r = result

    print("Best hyperparameters:", best_r)
    
    # Check it on the validation split
    train_nn_regression(X_train_r, y_train_r, X_val_r, y_val_r, X_val_r, y_val_r, best_r["params"], name="Final_KerasNN_Regression_validation")

    # Merge train + validation
    X_final_r = np.concatenate([X_train_r, X_val_r], axis=0)
    y_final_r = np.concatenate([y_train_r, y_val_r], axis=0)

    train_nn_regression(X_final_r, y_final_r, None, None, X_test_r, y_test_r, best_r["params"])



if __name__ == "__main__":
    main()


