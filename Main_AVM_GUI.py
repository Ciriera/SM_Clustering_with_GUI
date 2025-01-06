import gradio as gr
import pandas as pd
import numpy as np
import io
import psutil
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, Birch
import fastcluster
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

warnings.filterwarnings("ignore")

###############################################
# 1) Yardımcı Fonksiyonlar
###############################################
def measure_cpu_time():
    process = psutil.Process()
    return process.cpu_times().user

def load_data(uploaded_file):
    if not uploaded_file:
        raise ValueError("Hiç dosya yüklenmedi. Lütfen bir CSV dosyası seçin.")
    try:
        if hasattr(uploaded_file, 'name') and uploaded_file.name is not None:
            data = pd.read_csv(uploaded_file.name)
        else:
            data = pd.read_csv(io.BytesIO(uploaded_file.read()))
    except Exception as e:
        raise ValueError(f"CSV okunamadı: {e}")
    if data.empty:
        raise ValueError("Yüklediğiniz CSV dosyası boş. Lütfen geçerli bir veri seti yükleyin.")
    return data

def preprocess_data(data, selected_columns):
    """
    Sadece kullanıcı tarafından seçilen sütunları kullanıyoruz.
    Ardından one-hot + scaling.
    """
    if not selected_columns:
        raise ValueError("Hiçbir sütun seçmediniz. Lütfen en az bir sütun seçin.")
    data_subset = data[selected_columns].copy()
    data_encoded = pd.get_dummies(data_subset, drop_first=True)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_encoded)
    return scaled_data

def run_kmeans(scaled_data, n_clusters=3):
    start_time = measure_cpu_time()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    labels = kmeans.predict(scaled_data)
    cpu_time = measure_cpu_time() - start_time
    return labels, cpu_time

def run_fastcluster(scaled_data, n_clusters=3):
    start_time = measure_cpu_time()
    linkage_matrix = fastcluster.linkage_vector(scaled_data, method="ward")
    clusters = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    cpu_time = measure_cpu_time() - start_time
    return clusters, cpu_time

def run_birch(scaled_data, n_clusters=3):
    """
    Birch outlier üretmeyen bir algoritma (küme sayısı kadar böler).
    """
    start_time = measure_cpu_time()
    birch_model = Birch(n_clusters=n_clusters)
    labels = birch_model.fit_predict(scaled_data)
    cpu_time = measure_cpu_time() - start_time
    return labels, cpu_time

def run_algorithm(algo, scaled_data, n_clusters, fastcluster_number, birch_clusters):
    """
    Seçilen algoritmaya göre modeli çalıştır.
    """
    if algo == "MiniBatch K-Means":
        labels, cpu_time = run_kmeans(scaled_data, n_clusters)
    elif algo == "Fastcluster":
        labels, cpu_time = run_fastcluster(scaled_data, fastcluster_number)
    elif algo == "Birch":
        labels, cpu_time = run_birch(scaled_data, birch_clusters)
    else:
        labels, cpu_time = np.array([]), 0
    return labels, cpu_time

def visualize_clusters(algorithm, labels, data, selected_category, chart_type, idx=1):
    viz_data = data.copy()
    viz_data["cluster"] = labels

    if selected_category:
        if 'category' not in viz_data.columns:
            raise ValueError("Veri setinde 'category' sütunu yok.")
        viz_data = viz_data[viz_data['category'] == selected_category]

    unique_clusters = set(viz_data["cluster"])
    if len(unique_clusters) <= 1:
        raise ValueError(
            f"{algorithm} tek bir cluster veya veri kalmadı; farklı parametreler deneyin."
        )

    if chart_type == "Isı Haritası":
        required_cols = {'shopping_mall', 'quantity', 'cluster'}
        if not required_cols.issubset(viz_data.columns):
            raise ValueError(f"Isı Haritası için {required_cols} sütunları gerekli.")
        heatmap_data = viz_data.pivot_table(
            values="quantity",
            index="shopping_mall",
            columns="cluster",
            aggfunc="sum",
            fill_value=0
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".0f")

        # Cluster içindeki kategorileri grafik dışında ve daha düzenli göstermek için
        cluster_categories = viz_data.groupby("cluster")['category'].unique()
        cluster_dominance = viz_data.groupby("cluster")['category'].apply(lambda x: x.mode()[0])
        categories_text = "\n\n".join([f"Cluster {cluster} (Dominant: {cluster_dominance[cluster]}):\n- " + "\n- ".join(categories) for cluster, categories in cluster_categories.items()])
        plt.figtext(0.95, 0.5, categories_text, wrap=True, horizontalalignment='left', fontsize=10, va='center')

        plt.title(f"{algorithm} - Cluster Bazında Satış Yoğunluğu")
        plt.tight_layout()
        filename = f"heatmap_{idx}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        return filename

    elif chart_type == "Pie Chart":
        required_cols = {'quantity', 'cluster'}
        if not required_cols.issubset(viz_data.columns):
            raise ValueError(f"Pie Chart için {required_cols} sütunları gerekli.")
        pie_data = viz_data.groupby("cluster")["quantity"].sum()
        plt.figure(figsize=(6, 6))
        pie_data.plot.pie(autopct='%1.1f%%')
        plt.title(f"{algorithm} - Cluster Bazında Toplam Satışlar")
        plt.ylabel("")
        plt.tight_layout()
        filename = f"pie_chart_{idx}.png"
        plt.savefig(filename)
        plt.close()

        return filename

    elif chart_type == "Çubuk Grafiği":
        required_cols = {'quantity', 'cluster'}
        if not required_cols.issubset(viz_data.columns):
            raise ValueError(f"Çubuk Grafiği için {required_cols} sütunları gerekli.")
        bar_data = viz_data.groupby("cluster")["quantity"].sum()
        plt.figure(figsize=(8, 5))
        bar_data.plot(kind="bar")
        plt.title(f"{algorithm} - Cluster Bazında Toplam Satışlar")
        plt.tight_layout()
        filename = f"bar_chart_{idx}.png"
        plt.savefig(filename)
        plt.close()

        return filename

    return None

def hyperparameter_search(algo, scaled_data):
    best_params = None
    best_score = -1

    if algo == "MiniBatch K-Means":
        for k in range(2, 9):
            labels, _ = run_kmeans(scaled_data, k)
            sil = silhouette_score(scaled_data, labels)
            if sil > best_score:
                best_score = sil
                best_params = {'n_clusters': k}

    elif algo == "Fastcluster":
        for k in range(2, 9):
            labels, _ = run_fastcluster(scaled_data, k)
            sil = silhouette_score(scaled_data, labels)
            if sil > best_score:
                best_score = sil
                best_params = {'n_clusters': k}

    elif algo == "Birch":
        for k in range(2, 9):
            labels, _ = run_birch(scaled_data, k)
            sil = silhouette_score(scaled_data, labels)
            if sil > best_score:
                best_score = sil
                best_params = {'n_clusters': k}

    return best_params

###############################################
# 2) ANA İŞLEV: KÜMELEME + KARŞILAŞTIRMA
###############################################
def clustering_workflow(
    file,
    tek_algoritma,
    n_clusters,
    fastcluster_number,
    birch_clusters,
    selected_category,
    chart_type,
    algorithm_1,
    algorithm_2,
    optimize_algo1,
    optimize_algo2,
    selected_columns
):
    try:
        # Veri yükleme
        data = load_data(file)
        # Sadece seçilen sütunlarla ön işleme
        scaled_data = preprocess_data(data, selected_columns)

        # Hiperparametre
        best_params_algo1 = None
        if optimize_algo1 and algorithm_1 != "No Algorithm":
            best_params_algo1 = hyperparameter_search(algorithm_1, scaled_data)
        best_params_algo2 = None
        if optimize_algo2 and algorithm_2 != "No Algorithm":
            best_params_algo2 = hyperparameter_search(algorithm_2, scaled_data)

        if algorithm_1 == "No Algorithm" and algorithm_2 == "No Algorithm":
            df_err = pd.DataFrame({"Hata": ["Hiçbir algoritma seçmediniz."]})
            return df_err, None, None, "<p style='color:red;'>Lütfen en az bir algoritma seçin.</p>"

        # Tek algoritma
        if tek_algoritma or algorithm_1 == "No Algorithm" or algorithm_2 == "No Algorithm":
            if algorithm_1 != "No Algorithm":
                final_algo = algorithm_1
                if best_params_algo1 and 'n_clusters' in best_params_algo1:
                    chosen_n = best_params_algo1['n_clusters']
                    labels, cpu_time = run_algorithm(final_algo, scaled_data, chosen_n, fastcluster_number, chosen_n)
                else:
                    labels, cpu_time = run_algorithm(final_algo, scaled_data, n_clusters, fastcluster_number, birch_clusters)
            else:
                final_algo = algorithm_2
                if best_params_algo2 and 'n_clusters' in best_params_algo2:
                    chosen_n = best_params_algo2['n_clusters']
                    labels, cpu_time = run_algorithm(final_algo, scaled_data, chosen_n, fastcluster_number, chosen_n)
                else:
                    labels, cpu_time = run_algorithm(final_algo, scaled_data, n_clusters, fastcluster_number, birch_clusters)

            if labels.size == 0:
                df_err = pd.DataFrame({"Hata": ["Küme oluşturulamadı veya algoritma seçilmedi."]})
                return df_err, None, None, "<p style='color:red;'>Küme oluşturulamadı.</p>"

            sil_ = silhouette_score(scaled_data, labels)
            db_ = davies_bouldin_score(scaled_data, labels)
            ch_ = calinski_harabasz_score(scaled_data, labels)

            results_df = pd.DataFrame({
                "Metrik": ["Silhouette Score", "Davies-Bouldin Score", "Calinski-Harabasz Score", "CPU Time (s)"],
                f"{final_algo}": [sil_, db_, ch_, cpu_time]
            })

            chart1 = visualize_clusters(final_algo, labels, data, selected_category, chart_type, idx=1)
            chart2 = None

            # Arka plan #1f2937, ortalanmış
            comparison_html = f"""
            <div style='background: #1f2937; 
                        padding:15px; border-radius:8px; margin-top:10px; text-align:center;'>
                <h3 style='color:#fff; margin-bottom:10px;'>Tek Algoritma Sonuçları: {final_algo}</h3>
                <p style='color:#fff;'><strong>Silhouette:</strong> {sil_:.3f}</p>
                <p style='color:#fff;'><strong>Davies-Bouldin:</strong> {db_:.3f}</p>
                <p style='color:#fff;'><strong>Calinski-Harabasz:</strong> {ch_:.3f}</p>
                <p style='color:#fff;'><strong>CPU Time:</strong> {cpu_time:.3f} s</p>
            </div>
            """
            return results_df, chart1, chart2, comparison_html

        else:
            # Çift algoritma
            if best_params_algo1 and 'n_clusters' in best_params_algo1:
                labels1, cpu1 = run_algorithm(algorithm_1, scaled_data, best_params_algo1['n_clusters'], fastcluster_number, best_params_algo1['n_clusters'])
            else:
                labels1, cpu1 = run_algorithm(algorithm_1, scaled_data, n_clusters, fastcluster_number, birch_clusters)

            sil1 = silhouette_score(scaled_data, labels1)
            db1 = davies_bouldin_score(scaled_data, labels1)
            ch1 = calinski_harabasz_score(scaled_data, labels1)

            if best_params_algo2 and 'n_clusters' in best_params_algo2:
                labels2, cpu2 = run_algorithm(algorithm_2, scaled_data, best_params_algo2['n_clusters'], fastcluster_number, best_params_algo2['n_clusters'])
            else:
                labels2, cpu2 = run_algorithm(algorithm_2, scaled_data, n_clusters, fastcluster_number, birch_clusters)

            sil2 = silhouette_score(scaled_data, labels2)
            db2 = davies_bouldin_score(scaled_data, labels2)
            ch2 = calinski_harabasz_score(scaled_data, labels2)

            results_df = pd.DataFrame({
                "Metrik": ["Silhouette Score", "Davies-Bouldin Score", "Calinski-Harabasz Score", "CPU Time (s)"],
                f"{algorithm_1}": [sil1, db1, ch1, cpu1],
                f"{algorithm_2}": [sil2, db2, ch2, cpu2]
            })

            chart1 = visualize_clusters(algorithm_1, labels1, data, selected_category, chart_type, idx=1)
            chart2 = visualize_clusters(algorithm_2, labels2, data, selected_category, chart_type, idx=2)

            def metric_compare(m1, m2, bigger_is_better=True):
                if bigger_is_better:
                    if m1 > m2:
                        return f"{algorithm_1}"
                    elif m2 > m1:
                        return f"{algorithm_2}"
                    else:
                        return "Eşit"
                else:
                    if m1 < m2:
                        return f"{algorithm_1}"
                    elif m2 < m1:
                        return f"{algorithm_2}"
                    else:
                        return "Eşit"

            best_sil = metric_compare(sil1, sil2, True)
            best_db = metric_compare(db1, db2, False)
            best_ch = metric_compare(ch1, ch2, True)

            # Arka plan #1f2937, ortalanmış
            comparison_html = f"""
            <div style='background: #1f2937; 
                        padding:15px; border-radius:8px; margin-top:10px; text-align:center;'>
                <h3 style='color:#fff; margin-bottom:10px;'>Karşılaştırma Sonuçları</h3>

                <p style='color:#fff; margin:6px 0;'>
                    <strong>Silhouette:</strong><br/>
                    {algorithm_1}={sil1:.3f}, {algorithm_2}={sil2:.3f}
                    <br/><em>Daha iyi:</em> {best_sil}
                </p>
                <p style='color:#fff; margin:6px 0;'>
                    <strong>Davies-Bouldin:</strong><br/>
                    {algorithm_1}={db1:.3f}, {algorithm_2}={db2:.3f}
                    <br/><em>Daha iyi:</em> {best_db}
                </p>
                <p style='color:#fff; margin:6px 0;'>
                    <strong>Calinski-Harabasz:</strong><br/>
                    {algorithm_1}={ch1:.3f}, {algorithm_2}={ch2:.3f}
                    <br/><em>Daha iyi:</em> {best_ch}
                </p>
                <p style='color:#fff; margin:6px 0;'>
                    <strong>CPU Time (s):</strong><br/>
                    {algorithm_1}={cpu1:.3f}, {algorithm_2}={cpu2:.3f}
                </p>
            </div>
            """
            return results_df, chart1, chart2, comparison_html

    except Exception as e:
        empty_df = pd.DataFrame({"Hata": [str(e)]})
        return empty_df, None, None, f"<p style='color:red;'>Hata: {str(e)}</p>"

def reset_inputs():
    """
    Sıfırlayınca default değerlere döner.
    """
    return (
        None,   # file
        False,  # tek_algoritma
        3,      # n_clusters
        3,      # fastcluster_number
        3,      # birch_clusters
        "",     # selected_category
        "Isı Haritası",        # chart_type
        "No Algorithm",        # algorithm_1
        "No Algorithm",        # algorithm_2
        False,                 # optimize_algo1
        False,                 # optimize_algo2
        []
    )

###############################################
# 3) Login Ekranı ve Ana Ekran
###############################################
def check_login(u, p):
    return (u == "admin" and p == "admin")

custom_css = """
/* Login ekranını tam ortalamak için */
#login-container {
    display: flex !important;
    justify-content: center !important; /* Yatayda ortala */
    align-items: center !important;     /* Dikeyde ortala */
    height: 100vh !important;          /* Tüm pencere yüksekliği */
    padding: 20px;
    box-sizing: border-box;            /* Padding dahil hesaplansın */
}

/* Login kutusu */
#login-box {
    width: 100%;
    max-width: 400px; 
    padding: 20px;
    border-radius: 12px;
    background: #2d2d2d;
    box-shadow: 0 0 20px rgba(0,0,0,0.2);
}

/* CheckboxGroup'u tek satırda göstermek için */
.horizontal-group .wrap {
    white-space: nowrap !important;   /* Tek satıra zorla */
    overflow-x: auto !important;      /* Yatay kaydırma aktif */
}
.horizontal-group .wrap .checkbox, .horizontal-group .wrap .checkbox-item {
    display: inline-block !important;
    margin-right: 15px !important;
}
"""

import gradio as gr

with gr.Blocks(title="Login + Ana Ekran", css=custom_css) as demo:

    logged_in = gr.State(False)

    # -- LOGIN EKRANI --
    with gr.Group(visible=True, elem_id="login-container") as login_container:
        with gr.Group(elem_id="login-box"):
            gr.HTML("<h2 style='text-align:center; color:#fff;'>Hoş Geldiniz</h2>")
            username = gr.Textbox(label="Kullanıcı Adı", value="")
            password = gr.Textbox(label="Şifre", type="password", value="")
            login_button = gr.Button("Giriş")

    # -- ANA EKRAN --
    with gr.Group(visible=False) as main_box:
        gr.HTML("<h1 style='text-align:center;'>Clustering Karşılaştırma Uygulaması</h1>")
        gr.HTML("<div style='height:15px;'></div>")

        gr.Markdown("""
        <h3>Kullanım Detayları</h3>
        <ul>
            <li><strong>MiniBatch K-Means</strong>: K-Means varyantı</li>
            <li><strong>Fastcluster</strong>: Hiyerarşik benzeri</li>
            <li><strong>Birch</strong>: Outlier üretmez, n_clusters kadar böler</li>
        </ul>
        """)

        gr.HTML("<div style='height:15px;'></div>")

        file_input = gr.File(label="CSV Dosyası Yükleyin")

        with gr.Row():
            preview_open_btn = gr.Button("Ön İzleme Aç")
            preview_close_btn = gr.Button("Ön İzleme Kapat")

        with gr.Row():
            with gr.Column(scale=1):
                preview_df = gr.Dataframe(label="Önizleme (İlk 5 Satır)")
            with gr.Column(scale=1):
                selected_columns = gr.CheckboxGroup(
                    choices=[], 
                    label="Kullanılacak Sütunlar",
                    value=[],
                    elem_classes=["horizontal-group"]  # Tek satırda "ip" gibi
                )

        tek_algoritma_checkbox = gr.Checkbox(label="Tek Algoritma", value=False)

        with gr.Row():
            n_clusters = gr.Slider(2, 20, step=1, value=3, label="K-Means Cluster Sayısı")
            fastcluster_number = gr.Slider(2, 20, step=1, value=3, label="Fastcluster Cluster Sayısı")
            birch_clusters = gr.Slider(2, 20, step=1, value=3, label="Birch Cluster Sayısı")

        with gr.Row():
            selected_category = gr.Textbox(label="İstenen Kategori (Opsiyonel)")
            chart_type = gr.Dropdown(
                choices=["Isı Haritası", "Pie Chart", "Çubuk Grafiği"], 
                value="Isı Haritası", 
                label="Grafik Türü"
            )

        algorithm_choices = ["No Algorithm", "MiniBatch K-Means", "Fastcluster", "Birch"]
        with gr.Row():
            algorithm_1 = gr.Dropdown(
                choices=algorithm_choices, 
                value="No Algorithm", 
                label="Birinci Algoritma"
            )
            algorithm_2 = gr.Dropdown(
                choices=algorithm_choices, 
                value="No Algorithm", 
                label="İkinci Algoritma"
            )

        with gr.Row():
            optimize_algo1 = gr.Checkbox(label="Birinci Algoritma için Hiperparametre Optimizasyonu", value=False)
            optimize_algo2 = gr.Checkbox(label="İkinci Algoritma için Hiperparametre Optimizasyonu", value=False)

        with gr.Row():
            run_button = gr.Button("Çalıştır")
            reset_button = gr.Button("Sıfırla")

        results_table = gr.Dataframe(label="Sonuç Tablosu (Metrik Değerleri)")
        chart_out1 = gr.Image(label="Birinci Algoritma Grafiği")
        chart_out2 = gr.Image(label="İkinci Algoritma Grafiği")
        comparison_box = gr.HTML(label="Karşılaştırma Sonuçları")

        # Önizleme Aç/Kapat
        def file_preview_open(file, current_selection):
            try:
                df = load_data(file)
                cols = list(df.columns)
                # Tüm sütunları varsayılan olarak seçili gösterelim (value=cols)
                # Kullanıcı isterse uncheck yapar
                return df.head(5), gr.update(choices=cols, value=cols)
            except Exception as e:
                return pd.DataFrame({"Hata": [str(e)]}), gr.update()

        def file_preview_close(current_selection):
            # Sadece önizlemeyi (tablo) sıfırlayalım, check seçimlerini koruyalım
            return pd.DataFrame(), gr.update(value=current_selection)

        preview_open_btn.click(
            fn=file_preview_open,
            inputs=[file_input, selected_columns],
            outputs=[preview_df, selected_columns]
        )
        preview_close_btn.click(
            fn=file_preview_close,
            inputs=[selected_columns],
            outputs=[preview_df, selected_columns]
        )

        # Çalıştır
        run_button.click(
            fn=clustering_workflow,
            inputs=[
                file_input,
                tek_algoritma_checkbox,
                n_clusters,
                fastcluster_number,
                birch_clusters,
                selected_category,
                chart_type,
                algorithm_1,
                algorithm_2,
                optimize_algo1,
                optimize_algo2,
                selected_columns
            ],
            outputs=[
                results_table,
                chart_out1,
                chart_out2,
                comparison_box
            ]
        )

        # Sıfırla
        reset_button.click(
            fn=reset_inputs,
            inputs=[],
            outputs=[
                file_input,
                tek_algoritma_checkbox,
                n_clusters,
                fastcluster_number,
                birch_clusters,
                selected_category,
                chart_type,
                algorithm_1,
                algorithm_2,
                optimize_algo1,
                optimize_algo2,
                selected_columns
            ]
        )

        # Tek Algoritma Otomatik
        def auto_toggle_tek_algoritma(algo1, algo2):
            if algo1 == "No Algorithm" and algo2 == "No Algorithm":
                return True
            if algo1 == "No Algorithm" and algo2 != "No Algorithm":
                return True
            if algo2 == "No Algorithm" and algo1 != "No Algorithm":
                return True
            return False

        algorithm_1.change(
            fn=auto_toggle_tek_algoritma,
            inputs=[algorithm_1, algorithm_2],
            outputs=tek_algoritma_checkbox
        )
        algorithm_2.change(
            fn=auto_toggle_tek_algoritma,
            inputs=[algorithm_1, algorithm_2],
            outputs=tek_algoritma_checkbox
        )

    # LOGIN Fonksiyonları
    def do_login(u, p):
        return bool(check_login(u, p))

    login_button.click(
        fn=do_login,
        inputs=[username, password],
        outputs=[logged_in]
    )

    def update_visibility(is_logged):
        if is_logged:
            # login_container kapanıyor, main_box açılıyor
            return gr.update(visible=False), gr.update(visible=True)
        else:
            # login_container açılıyor, main_box kapanıyor
            return gr.update(visible=True), gr.update(visible=False)

    logged_in.change(
        fn=update_visibility,
        inputs=[logged_in],
        outputs=[login_container, main_box]
    )

demo.launch()
