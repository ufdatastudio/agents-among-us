import pandas as pd
import os
import joblib
from lime.lime_text import LimeTextExplainer
import warnings
warnings.filterwarnings("ignore")
def get_averaged_features(csv_paths, top_n=10):
    """Reads multiple CSVs, averages their weights, and splits by role."""
    valid_paths = [p for p in csv_paths if os.path.exists(p)]
    if not valid_paths:
        return [], []
    dfs = [pd.read_csv(p) for p in valid_paths]
    combined_df = pd.concat(dfs)
    avg_df = combined_df.groupby(['Feature', 'Correlated_Role'], as_index=False)['Weight'].mean()

    # Sort into Imposter and Crewmate
    imp = avg_df[avg_df['Correlated_Role'] == 'Imposter'].sort_values(by='Weight', ascending=False).to_dict('records')
    crew = avg_df[avg_df['Correlated_Role'] == 'Crewmate'].sort_values(by='Weight', ascending=True).to_dict('records')

    return imp[:top_n], crew[:top_n]

def get_mapped_label(label):
    """Maps the dataset's H/B to C/I for display."""
    if pd.isna(label): return "Unknown"
    val = str(label).upper().strip()
    if val == 'H': return 'C' # Honest -> Crewmate
    if val == 'B': return 'I' # Byzantine -> Imposter
    return val

def lime_explanations(models_dict, dataset_path, text_col, reported_col, stmt_col, label_col, output_file="results/lime_visualizations/combined_classifiers_report.html"):
    print("\n" + "="*95)
    print(f"{'GENERATING COMBINED LIME REPORT (4 Correct, 1 Incorrect)':^95}")
    print("="*95 + "\n")

    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset not found at {dataset_path}")
        return
        
    df_all = pd.read_csv(dataset_path)
    df_all = df_all.dropna(subset=[text_col, label_col]).copy()
    
    explainer = LimeTextExplainer(class_names=["Crewmate", "Imposter"])
    
    master_html = """
    <html>
    <head>
        <title>Agents Among Us: LIME Explanations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; }
            h1 { text-align: center; }
            h2 { color: #333; border-bottom: 2px solid #ccc; padding-bottom: 5px; margin-top: 50px;}
            .explanation-block { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 30px; }
            .meta-info { font-size: 1.0em; margin-bottom: 15px; padding: 10px; background: #eee; border-left: 4px solid #333;}
            .correct { border-left-color: #2e7d32; }
            .incorrect { border-left-color: #c62828; }
        </style>
    </head>
    <body>
        <h1>Agents Among Us: Classifier Feature Explanations</h1>
    """

    for model_name, model_path in models_dict.items():
        if not os.path.exists(model_path):
            print(f"[Warning] Model not found: {model_path}. Skipping.")
            continue
            
        print(f"\nProcessing Model: {model_name}")
        pipeline = joblib.load(model_path)
        master_html += f"<h2>Model: {model_name}</h2>\n"

        # 1. Take a pool of data to find correct/incorrect predictions
        df_pool = df_all.sample(n=min(1000, len(df_all)), random_state=42).copy()
        df_predict_pool = df_pool[[text_col, reported_col, stmt_col]]
        
        # Predict the outcomes (These will be 0s and 1s)
        preds_numeric = pipeline.predict(df_predict_pool)

        df_pool['Prediction'] = ['B' if p == 1 else 'H' for p in preds_numeric]
        
        # Now the string-to-string comparison works perfectly
        correct_mask = df_pool['Prediction'] == df_pool[label_col]
        
        #df_correct = df_pool[correct_mask & (df_pool[label_col] == 'B')]
        df_correct = df_pool[correct_mask]
        df_incorrect = df_pool[~correct_mask]
        
        n_corr = min(4, len(df_correct))
        n_inc = min(1, len(df_incorrect))
        
        print(f"  -> Found {len(df_correct)} correct and {len(df_incorrect)} incorrect in pool.")
        
        samples_correct = df_correct.sample(n=n_corr, random_state=43)
        samples_incorrect = df_incorrect.sample(n=n_inc, random_state=43)
        
        # Combine them and shuffle
        df_eval = pd.concat([samples_correct, samples_incorrect]).sample(frac=1, random_state=42)

        # 2. Generate explanations
        for idx, row in df_eval.iterrows():
            text_val = str(row[text_col])
            rep_val = row[reported_col]
            stmt_val = row[stmt_col]
            true_lbl_raw = row[label_col]
            pred_lbl_raw = row['Prediction']
            
            true_display = get_mapped_label(true_lbl_raw)
            pred_display = get_mapped_label(pred_lbl_raw)
            is_correct = (true_lbl_raw == pred_lbl_raw)
            
            def make_predictor(fixed_rep, fixed_stmt):
                def predictor(texts):
                    df_predict = pd.DataFrame({
                        text_col: texts,
                        reported_col: [fixed_rep] * len(texts),
                        stmt_col: [fixed_stmt] * len(texts)
                    })
                    try:
                        return pipeline.predict_proba(df_predict)
                    except AttributeError:
                        decisions = pipeline.decision_function(df_predict)
                        probs = expit(decisions)
                        return np.vstack([1 - probs, probs]).T
                return predictor

            print(f"  -> Explaining row {idx} (True: {true_display}, Pred: {pred_display})")
            custom_predictor = make_predictor(rep_val, stmt_val)
            
            exp = explainer.explain_instance(text_val, custom_predictor, num_features=6)
            
            raw_html = exp.as_html()
              
            status_class = "correct" if is_correct else "incorrect"
            status_text = "Correct Match" if is_correct else "Incorrect Prediction"
            
            master_html += f"""
            <div class="explanation-block">
                <div class="meta-info {status_class}">
                    <strong>True Label:</strong> {true_display} &nbsp;|&nbsp; 
                    <strong>Predicted Label:</strong> {pred_display} 
                    <span style="float:right;"><em>{status_text}</em></span><br><br>
                    <strong>Original Text:</strong> "{text_val}" <br>
                    <strong>Structural Features:</strong> Reported={rep_val}, Statement_Num={stmt_val}
                </div>
                {raw_html}
            </div>
            """
            
    master_html += "</body></html>"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(master_html)
        
if __name__ == "__main__":
    
    base_dir = "results/classifiers/features"
    # sgd_agg = os.path.join(base_dir, "sgd_classifier_aggregated_phrases.csv")
    # svm_agg = os.path.join(base_dir, "svm_aggregated_phrases.csv")
    
    # LIME Visualizations ---
    models = {
        "SVM": "results/classifiers/models_ngram/svm.joblib",
        "SGD": "results/classifiers/models_ngram/sgd.joblib"
    }

    dataset = "results/classifiers/data/observer_dataset.csv" 
    
    text_column_name = 'Text' 
    reported_column_name = 'Reported'
    statement_num_column_name = 'Statement_Num'
    true_label_column_name = 'Role' 
    
    lime_explanations(
        models_dict=models, 
        dataset_path=dataset, 
        text_col=text_column_name,
        reported_col=reported_column_name,
        stmt_col=statement_num_column_name,
        label_col=true_label_column_name,
        output_file="results/lime_visualizations/combined_classifiers_report.html"
    )