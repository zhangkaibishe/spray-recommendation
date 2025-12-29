
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置页面
st.set_page_config(
    page_title="消静电喷雾推荐系统",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 生成虚拟数据函数（备用）
@st.cache_data
def generate_spray_comparison_data(num_samples=600):
    """
    生成A、B两种喷雾的对比虚拟数据
    当真实数据不可用时使用
    """
    materials = ['棉', '涤纶', '羊毛', '尼龙', '防静电服']  # 修改：羊毛、尼龙
    spray_types = ['A型喷雾', 'B型喷雾']
    
    material_descriptions = {
        '棉': '100%纯棉面料',
        '涤纶': '100%涤纶织物', 
        '羊毛': '纯羊毛面料',  # 修改
        '尼龙': '100%尼龙面料',  # 修改
        '防静电服': '防静电专用面料'
    }
    
    data = []
    
    for _ in range(num_samples):
        material = np.random.choice(materials)
        spray_type = np.random.choice(spray_types)
        
        temperature = np.random.uniform(15, 35)
        humidity = np.random.uniform(30, 80)
        spray_volume = np.random.uniform(3, 8)
        
        base_performance = 0
        
        if material == '棉':
            base_performance = 82 if spray_type == 'A型喷雾' else 80
        elif material == '涤纶':
            base_performance = 78 if spray_type == 'A型喷雾' else 75
        elif material == '羊毛':  # 修改
            base_performance = 70 if spray_type == 'A型喷雾' else 88  # B型对羊毛更好
        elif material == '尼龙':  # 修改
            base_performance = 65 if spray_type == 'A型喷雾' else 85  # B型对尼龙更好
        else:  # 防静电服
            base_performance = 90 if spray_type == 'A型喷雾' else 89
        
        temp_effect = (25 - abs(temperature - 25)) * 0.3
        humidity_effect = (50 - abs(humidity - 50)) * 0.2
        volume_effect = (spray_volume - 3) * 1.5
        
        effectiveness = (base_performance + temp_effect + humidity_effect + 
                        volume_effect + np.random.normal(0, 3))
        effectiveness = np.clip(effectiveness, 0, 100)
        
        initial_resistance = 10 ** np.random.uniform(10, 12)
        resistance_reduction = effectiveness / 20
        after_resistance = initial_resistance / (10 ** (resistance_reduction * 0.1))
        
        decay_time = max(0.5, 5 - (effectiveness / 25) + np.random.normal(0, 0.3))
        duration = max(30, effectiveness * 2 + np.random.normal(0, 20))
        
        if effectiveness >= 85:
            phenomena = '静电完全消除，效果显著'
        elif effectiveness >= 75:
            phenomena = '静电明显消除，手感顺滑'
        elif effectiveness >= 65:
            phenomena = '静电部分消除，效果良好'
        else:
            phenomena = '静电消除效果一般'
        
        data.append({
            '材质类型': material,
            '材质详细描述': material_descriptions[material],
            '环境温度': round(temperature, 1),
            '环境湿度': round(humidity, 1),
            '消静电喷雾型号': spray_type,
            '喷雾用量': round(spray_volume, 1),
            '初始表面电阻': round(initial_resistance, 2),
            '喷雾后表面电阻': round(after_resistance, 2),
            '电荷衰减时间': round(decay_time, 2),
            '效果持续时间': round(duration, 1),
            '效果评分': round(effectiveness, 1),
            '实验现象与备注': phenomena
        })
    
    return pd.DataFrame(data)

# 数据清洗函数
def clean_experiment_data(df):
    """
    清洗实验数据，确保格式一致
    """
    cleaned_df = df.copy()
    
    column_mapping = {
        '材质': '材质类型',
        '材质类型': '材质类型',
        '喷雾类型': '消静电喷雾型号',
        '消静电喷雾型号': '消静电喷雾型号',
        '温度': '环境温度',
        '环境温度(℃)': '环境温度',
        '环境温度': '环境温度',
        '湿度': '环境湿度',
        '环境湿度(%RH)': '环境湿度',
        '环境湿度': '环境湿度',
        '用量': '喷雾用量',
        '喷雾用量(ml)': '喷雾用量',
        '喷雾用量': '喷雾用量',
        '初始电阻': '初始表面电阻',
        '初始表面电阻(Ω)': '初始表面电阻',
        '初始表面电阻': '初始表面电阻',
        '喷雾后电阻': '喷雾后表面电阻',
        '喷雾后表面电阻(Ω)': '喷雾后表面电阻',
        '喷雾后表面电阻': '喷雾后表面电阻',
        '衰减时间': '电荷衰减时间',
        '电荷衰减时间(s)': '电荷衰减时间',
        '电荷衰减时间': '电荷衰减时间',
        '持续时间': '效果持续时间',
        '效果持续时间(min)': '效果持续时间',
        '效果持续时间': '效果持续时间',
        '评分': '效果评分',
        '效果评分': '效果评分',
        '实验现象': '实验现象与备注',
        '实验现象与备注': '实验现象与备注',
        '备注': '实验现象与备注'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in cleaned_df.columns and new_name not in cleaned_df.columns:
            cleaned_df = cleaned_df.rename(columns={old_name: new_name})
    
    required_columns = ['材质类型', '消静电喷雾型号', '环境温度', '环境湿度']
    for col in required_columns:
        if col not in cleaned_df.columns:
            raise ValueError(f"数据中缺少必要的列: {col}")
    
    numeric_columns = ['环境温度', '环境湿度', '喷雾用量', 
                      '初始表面电阻', '喷雾后表面电阻', 
                      '电荷衰减时间', '效果持续时间']
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            if cleaned_df[col].isnull().sum() > 0:
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
    
    if '消静电喷雾型号' in cleaned_df.columns:
        cleaned_df['消静电喷雾型号'] = cleaned_df['消静电喷雾型号'].replace({
            'A': 'A型喷雾', 'B': 'B型喷雾',
            'a': 'A型喷雾', 'b': 'B型喷雾'
        })
        cleaned_df = cleaned_df[cleaned_df['消静电喷雾型号'].isin(['A型喷雾', 'B型喷雾'])]
    
    material_mapping = {
        '棉': '棉', '棉质': '棉', '棉质类': '棉',
        '涤纶': '涤纶', '化纤': '涤纶', '化纤类': '涤纶',
        '羊毛': '羊毛', '羊毛袜': '羊毛', '羊毛类': '羊毛',  # 修改：羊毛袜映射到羊毛
        '尼龙': '尼龙', '尼龙袜': '尼龙', '尼龙类': '尼龙',  # 修改：尼龙袜映射到尼龙
        '防静电服': '防静电服', '防静电': '防静电服', '特殊类': '防静电服'
    }
    
    if '材质类型' in cleaned_df.columns:
        cleaned_df['材质类型'] = cleaned_df['材质类型'].replace(material_mapping)
    
    return cleaned_df

# 计算效果评分函数
def calculate_effectiveness_scores(df):
    data = df.copy()
    
    numeric_cols = ['环境温度', '环境湿度', '喷雾用量',
                   '初始表面电阻', '喷雾后表面电阻',
                   '电荷衰减时间', '效果持续时间']
    
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    def calculate_row_score(row):
        score = 0
        weights = []
        
        if pd.notna(row.get('初始表面电阻')) and pd.notna(row.get('喷雾后表面电阻')):
            if row['喷雾后表面电阻'] > 0:
                resistance_ratio = row['初始表面电阻'] / row['喷雾后表面电阻']
                resistance_score = min(100, np.log10(max(resistance_ratio, 1)) * 25)
                score += resistance_score * 0.4
                weights.append(0.4)
        
        if pd.notna(row.get('电荷衰减时间')):
            decay_score = max(0, 100 - row['电荷衰减时间'] * 10)
            score += decay_score * 0.3
            weights.append(0.3)
        
        if pd.notna(row.get('效果持续时间')):
            duration_score = min(100, row['效果持续时间'] / 3)
            score += duration_score * 0.3
            weights.append(0.3)
        
        if weights:
            total_weight = sum(weights)
            if total_weight > 0:
                score = score / total_weight * 100
        
        return round(score, 1)
    
    data['效果评分'] = data.apply(calculate_row_score, axis=1)
    
    return data

# 加载真实实验数据
@st.cache_data
def load_real_experiment_data(file_path):
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        else:
            raise ValueError("不支持的文件格式，请使用Excel或CSV文件")
        
        st.success(f"成功读取文件: {os.path.basename(file_path)}")
        st.info(f"原始数据行数: {len(df)}")
        
        df = clean_experiment_data(df)
        st.info(f"清洗后数据行数: {len(df)}")
        
        if '效果评分' not in df.columns:
            df = calculate_effectiveness_scores(df)
            st.info("已计算效果评分")
        
        missing_data = df.isnull().sum().sum()
        if missing_data > 0:
            st.warning(f"数据中存在 {missing_data} 个缺失值，已使用中位数填充")
        
        return df
    
    except Exception as e:
        st.error(f"加载实验数据失败: {str(e)}")
        return None

# 推荐系统类
class SprayRecommendationSystem:
    def __init__(self, data):
        self.data = data
        self.materials = ['棉', '涤纶', '羊毛', '尼龙', '防静电服']  # 修改
        self.model = None
        self.feature_columns = None
        
    def train_recommendation_model(self):
        training_data = []
        
        for _, row in self.data.iterrows():
            material = row['材质类型']
            temperature = row['环境温度']
            humidity = row['环境湿度']
            
            condition_data = self.data[
                (self.data['材质类型'] == material) & 
                (self.data['环境温度'].between(temperature-2, temperature+2)) &
                (self.data['环境湿度'].between(humidity-5, humidity+5))
            ]
            
            if len(condition_data) >= 2:
                a_effect = condition_data[condition_data['消静电喷雾型号'] == 'A型喷雾']['效果评分'].mean()
                b_effect = condition_data[condition_data['消静电喷雾型号'] == 'B型喷雾']['效果评分'].mean()
                
                best_spray = 'A型喷雾' if a_effect > b_effect else 'B型喷雾'
                
                training_data.append({
                    '材质类型': material,
                    '环境温度': temperature,
                    '环境湿度': humidity,
                    '最佳喷雾': best_spray
                })
        
        training_df = pd.DataFrame(training_data)
        
        if training_df.empty:
            return None
        
        X = pd.get_dummies(training_df[['材质类型', '环境温度', '环境湿度']], 
                          columns=['材质类型'])
        y = training_df['最佳喷雾']
        
        self.feature_columns = X.columns.tolist()
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        return self.model
    
    def recommend_spray(self, material, temperature=25, humidity=50):
        if material not in self.materials:
            return f"错误: 不支持的材质类型 '{material}'，请选择: {self.materials}"
        
        if self.model is None:
            return self._rule_based_recommendation(material, temperature, humidity)
        
        try:
            input_features = {
                '环境温度': temperature,
                '环境湿度': humidity
            }
            
            for m in self.materials:
                input_features[f'材质类型_{m}'] = 1 if m == material else 0
            
            input_df = pd.DataFrame([input_features])[self.feature_columns]
            
            recommended_spray = self.model.predict(input_df)[0]
            
            reasoning = self._get_recommendation_reasoning(material, temperature, humidity, recommended_spray)
            
            return {
                '推荐结果': recommended_spray,
                '输入条件': {
                    '材质类型': material,
                    '环境温度': f"{temperature}℃",
                    '环境湿度': f"{humidity}%RH"
                },
                '推荐依据': reasoning,
                '使用建议': self._get_usage_suggestion(material, recommended_spray)
            }
            
        except Exception as e:
            return self._rule_based_recommendation(material, temperature, humidity)
    
    def _rule_based_recommendation(self, material, temperature, humidity):
        # 修改：判断羊毛和尼龙
        if material in ['羊毛', '尼龙']:
            recommended_spray = 'B型喷雾'
            reason = f"实验数据显示B型喷雾对{material}有更好的消静电效果"
        elif material in ['棉', '涤纶']:
            recommended_spray = 'A型喷雾'
            reason = f"A型喷雾对{material}材质的适应性更好"
        else:
            if humidity < 40:
                recommended_spray = 'B型喷雾'
                reason = "干燥环境下B型喷雾表现更稳定"
            else:
                recommended_spray = 'A型喷雾'
                reason = "正常湿度下A型喷雾效果良好"
        
        return {
            '推荐结果': recommended_spray,
            '输入条件': {
                '材质类型': material,
                '环境温度': f"{temperature}℃",
                '环境湿度': f"{humidity}%RH"
            },
            '推荐依据': reason,
            '使用建议': self._get_usage_suggestion(material, recommended_spray)
        }
    
    def _get_recommendation_reasoning(self, material, temperature, humidity, recommended_spray):
        material_data = self.data[self.data['材质类型'] == material]
        
        if recommended_spray == 'A型喷雾':
            comparison_spray = 'B型喷雾'
        else:
            comparison_spray = 'A型喷雾'
        
        recommended_avg = material_data[material_data['消静电喷雾型号'] == recommended_spray]['效果评分'].mean()
        comparison_avg = material_data[material_data['消静电喷雾型号'] == comparison_spray]['效果评分'].mean()
        
        improvement = recommended_avg - comparison_avg
        
        if improvement > 5:
            reason = f"实验数据显示{recommended_spray}对{material}的消静电效果明显优于{comparison_spray}"
        elif improvement > 2:
            reason = f"实验数据显示{recommended_spray}对{material}的消静电效果略优于{comparison_spray}"
        else:
            reason = f"在当前环境条件下，{recommended_spray}对{material}的适应性更好"
        
        return reason
    
    def _get_usage_suggestion(self, material, spray_type):
        suggestions = {
            '棉': {
                'A型喷雾': '建议用量5-7ml，均匀喷洒于表面',
                'B型喷雾': '建议用量4-6ml，注意通风使用'
            },
            '涤纶': {
                'A型喷雾': '建议用量6-8ml，喷洒后轻拍均匀',
                'B型喷雾': '建议用量5-7ml，避免过量使用'
            },
            '羊毛': {  # 修改
                'A型喷雾': '建议用量3-5ml，喷洒后自然风干',
                'B型喷雾': '建议用量4-6ml，对羊毛材质有更好亲和性'
            },
            '尼龙': {  # 修改
                'A型喷雾': '建议用量3-5ml，注意喷洒距离',
                'B型喷雾': '建议用量4-6ml，能有效降低尼龙静电'
            },
            '防静电服': {
                'A型喷雾': '建议用量8-10ml，全面均匀喷洒',
                'B型喷雾': '建议用量7-9ml，干燥环境下效果更佳'
            }
        }
        
        return suggestions.get(material, {}).get(spray_type, '建议用量5-7ml，均匀喷洒')

# 主应用
def main():
    st.title("🧪 多材质消静电喷雾推荐系统")
    st.markdown("---")
    
    st.sidebar.title("⚙️ 系统配置")
    
    data_source = st.sidebar.radio(
        "选择数据源",
        ["使用真实实验数据", "使用虚拟数据"]
    )
    
    df = None
    
    if data_source == "使用真实实验数据":
        st.sidebar.subheader("📁 数据上传")
        
        uploaded_file = st.sidebar.file_uploader(
            "上传实验数据文件",
            type=['xlsx', 'xls', 'csv'],
            help="支持Excel(.xlsx, .xls)和CSV格式"
        )
        
        if uploaded_file is not None:
            file_path = f"uploaded_data.{uploaded_file.name.split('.')[-1]}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("正在加载和分析实验数据..."):
                df = load_real_experiment_data(file_path)
                
                if df is not None:
                    st.sidebar.success(f"✅ 成功加载 {len(df)} 条实验记录")
                else:
                    st.sidebar.error("❌ 数据加载失败，将使用虚拟数据")
                    df = generate_spray_comparison_data(300)
        else:
            default_files = ['实验记录表.xlsx', '实验数据.xlsx', 'data.xlsx']
            data_loaded = False
            
            for file_name in default_files:
                if os.path.exists(file_name):
                    with st.spinner(f"正在加载 {file_name}..."):
                        df = load_real_experiment_data(file_name)
                        if df is not None:
                            st.sidebar.info(f"📂 已加载默认文件: {file_name}")
                            data_loaded = True
                            break
            
            if not data_loaded:
                st.sidebar.warning("⚠️ 未找到实验数据文件，请上传文件或使用虚拟数据")
                st.info("请上传实验数据文件或切换到虚拟数据模式")
                return
    
    else:
        st.sidebar.info("🔬 虚拟数据模式 - 用于演示和测试")
        df = generate_spray_comparison_data(300)
    
    if df is not None:
        with st.spinner("正在训练推荐模型..."):
            recommender = SprayRecommendationSystem(df)
            recommender.train_recommendation_model()
        
        st.sidebar.title("📋 系统导航")
        app_mode = st.sidebar.selectbox(
            "选择功能", 
            ["喷雾推荐", "数据可视化", "数据分析", "关于系统"]
        )
        
        if app_mode == "喷雾推荐":
            show_recommendation_interface(recommender)
        elif app_mode == "数据可视化":
            show_visualization_interface(df)
        elif app_mode == "数据分析":
            show_data_analysis_interface(df)
        else:
            show_about_interface()
        
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: gray;'>"
            "多材质消静电喷雾推荐系统 © 2024 毕业设计项目"
            "</div>", 
            unsafe_allow_html=True
        )

# 喷雾推荐界面
def show_recommendation_interface(recommender):
    st.header("🧴 消静电喷雾智能推荐")
    st.markdown("根据服装材质和环境条件，智能推荐最适合的消静电喷雾")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("输入条件")
        
        material = st.selectbox(
            "选择服装材质",
            options=['棉', '涤纶', '羊毛', '尼龙', '防静电服'],  # 修改
            help="选择需要消除静电的服装材质"
        )
        
        temperature = st.slider(
            "环境温度 (℃)",
            min_value=10,
            max_value=40,
            value=25,
            help="当前环境温度"
        )
        
        humidity = st.slider(
            "环境湿度 (%RH)",
            min_value=20,
            max_value=90,
            value=50,
            help="当前环境湿度"
        )
        
        recommend_button = st.button("获取推荐", type="primary")
    
    with col2:
        st.subheader("推荐结果")
        
        if recommend_button:
            with st.spinner("正在分析最佳喷雾..."):
                recommendation = recommender.recommend_spray(material, temperature, humidity)
            
            st.success(f"推荐使用: **{recommendation['推荐结果']}**")
            st.info(f"**推荐依据:** {recommendation['推荐依据']}")
            st.warning(f"**使用建议:** {recommendation['使用建议']}")
            st.markdown("**输入条件:**")
            st.json(recommendation['输入条件'])
        else:
            st.info("请选择材质和环境条件，然后点击'获取推荐'按钮")
            
            material_descriptions = {
                '棉': '纯棉材质容易产生静电，尤其在干燥环境下',
                '涤纶': '涤纶是合成纤维，静电问题较为常见',
                '羊毛': '羊毛材质在干燥条件下易产生静电',  # 修改
                '尼龙': '尼龙材质静电明显，需要专门处理',  # 修改
                '防静电服': '专业防静电服装，但仍需定期维护'
            }
            
            if material:
                st.markdown(f"**{material}材质特点:** {material_descriptions[material]}")

# 数据可视化界面
def show_visualization_interface(df):
    st.header("📊 实验数据可视化分析")
    
    tab1, tab2, tab3 = st.tabs(["效果对比", "环境影响", "数据统计"])
    
    with tab1:
        st.subheader("A/B喷雾效果对比")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='材质类型', y='效果评分', hue='消静电喷雾型号', ax=ax)
        ax.set_title('各材质A/B喷雾效果对比')
        ax.legend(title='喷雾型号')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.subheader("平均效果对比")
        effect_comparison = df.groupby(['材质类型', '消静电喷雾型号'])['效果评分'].mean().unstack()
        st.bar_chart(effect_comparison)
    
    with tab2:
        st.subheader("环境条件对效果的影响")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            for spray_type in ['A型喷雾', 'B型喷雾']:
                spray_data = df[df['消静电喷雾型号'] == spray_type]
                ax.scatter(spray_data['环境温度'], spray_data['效果评分'], 
                          alpha=0.6, label=spray_type)
            ax.set_xlabel('环境温度(℃)')
            ax.set_ylabel('效果评分')
            ax.set_title('温度对效果的影响')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            for spray_type in ['A型喷雾', 'B型喷雾']:
                spray_data = df[df['消静电喷雾型号'] == spray_type]
                ax.scatter(spray_data['环境湿度'], spray_data['效果评分'], 
                          alpha=0.6, label=spray_type)
            ax.set_xlabel('环境湿度(%RH)')
            ax.set_ylabel('效果评分')
            ax.set_title('湿度对效果的影响')
            ax.legend()
            st.pyplot(fig)
    
    with tab3:
        st.subheader("实验数据统计")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总实验次数", len(df))
        with col2:
            st.metric("涉及材质种类", df['材质类型'].nunique())
        with col3:
            st.metric("平均效果评分", f"{df['效果评分'].mean():.1f}")
        
        st.subheader("实验数据样本")
        st.dataframe(df.head(10), use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下载完整数据 (CSV)",
            data=csv,
            file_name="消静电喷雾实验数据.csv",
            mime="text/csv"
        )

# 数据分析界面
def show_data_analysis_interface(df):
    st.header("📈 实验数据分析")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("实验总数", len(df))
    with col2:
        st.metric("平均效果评分", f"{df['效果评分'].mean():.1f}")
    with col3:
        best_material = df.groupby('材质类型')['效果评分'].mean().idxmax()
        st.metric("最佳效果材质", best_material)
    
    st.subheader("🔍 数据质量检查")
    
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        st.warning(f"发现 {missing_data.sum()} 个缺失值")
        missing_df = pd.DataFrame({
            '列名': missing_data.index,
            '缺失数量': missing_data.values,
            '缺失比例': (missing_data.values / len(df) * 100).round(1)
        })
        st.dataframe(missing_df[missing_df['缺失数量'] > 0])
    else:
        st.success("✅ 数据完整，无缺失值")
    
    st.subheader("📊 A/B喷雾效果对比")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_data = df.groupby(['材质类型', '消静电喷雾型号'])['效果评分'].mean().unstack()
    comparison_data.plot(kind='bar', ax=ax)
    ax.set_ylabel('平均效果评分')
    ax.set_title('各材质A/B喷雾效果对比')
    ax.legend(title='喷雾型号')
    st.pyplot(fig)
    
    st.subheader("🔬 B型喷雾优化效果分析")
    
    key_materials = ['羊毛', '尼龙']  # 修改
    
    for material in key_materials:
        if material in df['材质类型'].unique():
            material_data = df[df['材质类型'] == material]
            
            if len(material_data) > 0:
                a_effect = material_data[material_data['消静电喷雾型号'] == 'A型喷雾']['效果评分'].mean()
                b_effect = material_data[material_data['消静电喷雾型号'] == 'B型喷雾']['效果评分'].mean()
                
                improvement = b_effect - a_effect
                improvement_pct = (improvement / a_effect * 100) if a_effect > 0 else 0
                
                st.write(f"**{material}**:")
                st.write(f"- A型喷雾平均效果: {a_effect:.1f}")
                st.write(f"- B型喷雾平均效果: {b_effect:.1f}")
                st.write(f"- B型喷雾提升: {improvement:.1f}分 ({improvement_pct:.1f}%)")
                
                if improvement > 0:
                    st.success(f"✅ B型喷雾对{material}有正面优化效果")
                else:
                    st.warning(f"⚠️ B型喷雾对{material}效果不明显")
                
                st.write("---")
    
    st.subheader("🌡️ 环境条件分析")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.scatter(df['环境温度'], df['效果评分'], alpha=0.6)
        ax.set_xlabel('环境温度(℃)')
        ax.set_ylabel('效果评分')
        ax.set_title('温度与效果关系')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots()
        ax.scatter(df['环境湿度'], df['效果评分'], alpha=0.6)
        ax.set_xlabel('环境湿度(%RH)')
        ax.set_ylabel('效果评分')
        ax.set_title('湿度与效果关系')
        st.pyplot(fig)
    
    st.subheader("📋 原始实验数据")
    show_raw_data = st.checkbox("显示原始数据")
    if show_raw_data:
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下载分析数据 (CSV)",
            data=csv,
            file_name="实验数据分析结果.csv",
            mime="text/csv"
        )

# 关于系统界面
def show_about_interface():
    st.header("ℹ️ 关于系统")
    
    st.markdown("""
    ### 系统介绍
    
    **多材质消静电喷雾推荐系统**是一个基于机器学习的智能推荐平台，旨在帮助用户根据不同的服装材质和环境条件选择最合适的消静电喷雾。
    
    ### 系统特点
    
    - **科学推荐**: 基于大量实验数据训练机器学习模型
    - **多维度考虑**: 综合考虑材质特性、环境温湿度等因素
    - **用户友好**: 简洁直观的界面，操作简单便捷
    - **数据驱动**: 所有推荐均基于实验数据和分析结果
    
    ### 喷雾类型说明
    
    - **A型喷雾**: 基础配方，对棉、涤纶等常见材质有良好效果
    - **B型喷雾**: 在A型基础上优化配方，对羊毛、尼龙等特殊材质有更好效果
    
    ### 使用方法
    
    1. 在"喷雾推荐"页面选择服装材质
    2. 设置当前环境温度和湿度
    3. 点击"获取推荐"按钮查看推荐结果
    4. 按照使用建议正确使用喷雾
    
    ### 技术实现
    
    - **数据分析**: Python, Pandas, NumPy
    - **机器学习**: Scikit-learn, 随机森林算法
    - **可视化**: Matplotlib, Seaborn
    - **Web界面**: Streamlit
    
    ### 开发信息
    
    本系统为本科毕业设计项目，专注于多材质消静电喷雾的智能推荐研究。
    
    ### 数据说明
    
    系统支持两种数据模式：
    1. **真实实验数据**: 上传您的实验数据Excel/CSV文件
    2. **虚拟数据**: 系统生成的模拟数据，用于演示和测试
    
    ### 注意事项
    
    - 确保实验数据格式正确
    - 推荐结果基于已有实验数据，新材质可能需要额外实验
    - 系统会不断优化，建议定期更新实验数据
    """)

if __name__ == "__main__":
    main()
