import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit.components.v1 as components

# ==========================================
# 1. 页面配置与样式
# ==========================================
st.set_page_config(
    page_title="贾宝玉社会网络演变分析",
    page_icon="🏮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS以增加学术感
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #8B0000; /* 深红色，呼应红楼梦 */
        font-family: "Serif";
    }
    h2, h3 {
        color: #333333;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据源定义
# ==========================================
# 定义三个阶段的数据链接
DATA_SOURCES = {
    "Phase 1: 天真少年 (19-23回)": {
        "edges": "https://raw.githubusercontent.com/seblee424/jiabaoyu_social_network/main/edges_phase1_%E5%A4%A9%E7%9C%9F%E5%B0%91%E5%B9%B4(19-23%E5%9B%9E).csv",
        "nodes": "https://raw.githubusercontent.com/seblee424/jiabaoyu_social_network/main/nodes_phase1_%E5%A4%A9%E7%9C%9F%E5%B0%91%E5%B9%B4(19-23%E5%9B%9E).csv",
        "desc": "此阶段贾宝玉生活在相对无忧无虑的大观园初期，试图建立纯洁的女儿国乌托邦。"
    },
    "Phase 2: 情感觉醒 (26-29回)": {
        "edges": "https://raw.githubusercontent.com/seblee424/jiabaoyu_social_network/main/edges_phase2_%E6%83%85%E6%84%9F%E8%A7%89%E9%86%92(26-29%E5%9B%9E).csv",
        "nodes": "https://raw.githubusercontent.com/seblee424/jiabaoyu_social_network/main/nodes_phase2_%E6%83%85%E6%84%9F%E8%A7%89%E9%86%92(26-29%E5%9B%9E).csv",
        "desc": "情感纠葛加深，宝黛钗三人的关系成为核心，社交网络开始显现情感张力。"
    },
    "Phase 3: 现实冲击 (32-36回)": {
        "edges": "https://raw.githubusercontent.com/seblee424/jiabaoyu_social_network/main/edges_phase3_%E7%8E%B0%E5%AE%9E%E5%86%B2%E5%87%BB(32-36%E5%9B%9E).csv",
        "nodes": "https://raw.githubusercontent.com/seblee424/jiabaoyu_social_network/main/nodes_phase3_%E7%8E%B0%E5%AE%9E%E5%86%B2%E5%87%BB(32-36%E5%9B%9E).csv",
        "desc": "金钏之死、宝玉挨打等事件发生，外部残酷现实打破了理想世界的宁静。"
    }
}

# ==========================================
# 3. 数据处理与计算函数
# ==========================================
@st.cache_data
def load_data(edges_url, nodes_url):
    """加载并清洗CSV数据"""
    try:
        edges_df = pd.read_csv(edges_url)
        nodes_df = pd.read_csv(nodes_url)
        return edges_df, nodes_df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None, None

def create_graph(edges_df, nodes_df):
    """构建NetworkX图对象并计算指标"""
    G = nx.Graph()
    
    # 1. 添加节点
    # 假设CSV中有 'Id', 'Label' 列，如果没有则需要根据实际CSV调整
    # 通常 Gephi 导出的 CSV 列名为 Id, Label, Source, Target, Weight
    for _, row in nodes_df.iterrows():
        # 兼容不同的列名写法
        node_id = row.get('Id') or row.get('id')
        label = row.get('Label') or row.get('label') or str(node_id)
        G.add_node(node_id, label=label, title=label)
        
    # 2. 添加边
    for _, row in edges_df.iterrows():
        src = row.get('Source') or row.get('source')
        tgt = row.get('Target') or row.get('target')
        w = row.get('Weight') or row.get('weight') or 1
        if src in G.nodes and tgt in G.nodes:
            G.add_edge(src, tgt, weight=w)
            
    return G

def calculate_metrics(G):
    """计算网络分析指标"""
    # 1. 网络密度
    density = nx.density(G)
    
    # 2. 度中心性 (Degree Centrality)
    degree_dict = nx.degree_centrality(G)
    
    # 3. 中介中心性 (Betweenness Centrality)
    betweenness_dict = nx.betweenness_centrality(G, weight='weight')
    
    # 4. 社群检测 (Louvain Modularity)
    partition = community_louvain.best_partition(G, weight='weight')
    modularity_score = community_louvain.modularity(partition, G)
    
    # 将指标存入节点属性
    nx.set_node_attributes(G, degree_dict, 'degree_centrality')
    nx.set_node_attributes(G, betweenness_dict, 'betweenness_centrality')
    nx.set_node_attributes(G, partition, 'group')
    
    return G, density, modularity_score, degree_dict, betweenness_dict, partition

# ==========================================
# 4. 可视化函数
# ==========================================
def visualize_network(G, partition):
    """使用PyVis生成交互式网络图"""
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # 根据社群分配颜色
    # 生成颜色映射
    unique_communities = list(set(partition.values()))
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_map = {com: colors[i % len(colors)] for i, com in enumerate(unique_communities)}
    
    for node_id in G.nodes:
        node = G.nodes[node_id]
        # 大小基于度中心性 * 一个系数，使其在图中可见
        size = node['degree_centrality'] * 30 + 10
        group_id = node['group']
        color = color_map.get(group_id, "#97C2FC")
        
        # Tooltip 显示详细信息
        title_html = f"<b>{node['label']}</b><br>Degree: {node['degree_centrality']:.3f}<br>Betweenness: {node['betweenness_centrality']:.3f}<br>Group: {group_id}"
        
        net.add_node(node_id, label=node['label'], title=title_html, size=size, color=color, group=group_id)

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        net.add_edge(u, v, value=weight, color="#cccccc")

    # 布局算法：使用 Force Atlas 2 (PyVis 中对应 force_atlas_2based)
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08,
        damping=0.4,
        overlap=0
    )
    
    # 保存为临时HTML并读取
    try:
        path = "/tmp"
        net.save_graph(f'pyvis_graph.html')
        HtmlFile = open(f'pyvis_graph.html', 'r', encoding='utf-8')
        return HtmlFile.read()
    except:
        # 本地运行 fallback
        net.save_graph('pyvis_graph.html')
        HtmlFile = open('pyvis_graph.html', 'r', encoding='utf-8')
        return HtmlFile.read()

# ==========================================
# 5. 主程序逻辑
# ==========================================
def main():
    # ---- Sidebar: 导航与设置 ----
    st.sidebar.title("📖 导航栏")
    st.sidebar.info("课程: CHC5904\n学生作业: Hands-on Assignment #2")
    
    selected_phase = st.sidebar.selectbox(
        "选择分析阶段",
        list(DATA_SOURCES.keys())
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Research Question")
    st.sidebar.markdown("""
    **如何通过社会网络的变化体现贾宝玉的个人成长？**
    
    本研究通过对比三个关键人生阶段（天真、觉醒、冲击）的社交网络结构，
    分析贾宝玉在《红楼梦》大观园体系中的位置变迁。
    """)
    
    # ---- Main Content ----
    st.title(f"🕸️ {selected_phase}")
    st.markdown(f"_{DATA_SOURCES[selected_phase]['desc']}_")
    
    # 加载数据
    with st.spinner('正在从GitHub加载数据并运行算法...'):
        edges_url = DATA_SOURCES[selected_phase]['edges']
        nodes_url = DATA_SOURCES[selected_phase]['nodes']
        
        edges_df, nodes_df = load_data(edges_url, nodes_url)
        
        if edges_df is not None and nodes_df is not None:
            # 构建图 & 计算
            G = create_graph(edges_df, nodes_df)
            G, density, modularity, degree, betweenness, partition = calculate_metrics(G)
            
            # 1. 展示关键指标 (Metrics Dashboard)
            st.subheader("📊 网络整体指标 (Network Metrics)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Nodes (节点数)", G.number_of_nodes())
            col2.metric("Edges (边数)", G.number_of_edges())
            col3.metric("Density (密度)", f"{density:.4f}")
            col4.metric("Modularity (模块化)", f"{modularity:.4f}")
            
            with st.expander("指标解释 (Metric Definitions)"):
                st.markdown("""
                - **Density**: 网络中实际连接数与可能的最大连接数之比。反映社交圈的紧密程度。
                - **Modularity**: 衡量网络划分成社群的好坏程度。值越高说明社群分化越明显。
                """)

            # 2. 交互式网络图 (Visualization)
            st.subheader("🕸️ 交互式网络可视化 (Interactive Visualization)")
            st.markdown("节点大小 = 度中心性 | 节点颜色 = 社群 (Community) | 布局 = Force Atlas 2")
            
            # 生成HTML
            html_data = visualize_network(G, partition)
            components.html(html_data, height=620)
            
            # 3. 详细数据分析 (Deep Dive)
            st.subheader("🔍 核心人物分析 (Centrality Analysis)")
            
            # 准备DataFrame用于展示
            metrics_df = pd.DataFrame({
                'Character': [G.nodes[n]['label'] for n in G.nodes],
                'Degree (影响力)': [degree[n] for n in G.nodes],
                'Betweenness (桥接能力)': [betweenness[n] for n in G.nodes],
                'Community (社群)': [partition[n] for n in G.nodes]
            }).sort_values(by='Degree (影响力)', ascending=False)
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**影响力排名 (Top by Degree)**")
                st.markdown("体现人物在当前社交圈中的活跃度和直接联系数量。")
                st.dataframe(metrics_df[['Character', 'Degree (影响力)']].head(10), use_container_width=True)
                
            with c2:
                st.markdown("**中介能力排名 (Top by Betweenness)**")
                st.markdown("体现人物作为信息“桥梁”的能力，控制着不同小圈子间的沟通。")
                st.dataframe(metrics_df.sort_values(by='Betweenness (桥接能力)', ascending=False)[['Character', 'Betweenness (桥接能力)']].head(10), use_container_width=True)
            
            # 4. 思考与结论 (Reflection Placeholder)
            st.markdown("---")
            st.subheader("📝 阶段性观察 (Reflection)")
            
            if "Phase 1" in selected_phase:
                st.info("**观察**: 在这个阶段，贾宝玉处于网络的绝对中心，网络密度可能较高，因为大观园主要人物都围绕着他。社群分化可能不明显。")
            elif "Phase 2" in selected_phase:
                st.info("**观察**: 随着情感觉醒，林黛玉和薛宝钗在网络中的权重可能会上升。注意观察宝玉的中介中心性变化，他是否开始在不同群体（如长辈vs同辈）间周旋？")
            else:
                st.info("**观察**: 现实冲击阶段。网络可能会出现断裂或重组。某些边缘人物（如袭人、王夫人）的重要性是否因这一阶段的冲突事件（如挨打）而上升？")

        else:
            st.warning("无法加载数据，请检查CSV文件格式或网络连接。")

if __name__ == "__main__":
    main()
