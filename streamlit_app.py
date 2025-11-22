import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit.components.v1 as components


st.set_page_config(
    page_title="贾宝玉社会网络演变分析",
    page_icon="🏮",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #8B0000; /* 深红色，呼应红楼梦 */
        font-family: "Serif";
    }
    h2, h3, h4 {
        color: #333333;
        font-family: "Serif";
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    /* 自定义结论框样式 */
    .conclusion-box {
        background-color: #fdf3f2; /* 淡淡的红色背景 */
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #8B0000;
        margin-top: 20px;
        font-size: 16px;
        line-height: 1.6;
    }
    .stage-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)


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

@st.cache_data
def load_data(edges_url, nodes_url):
    try:
        edges_df = pd.read_csv(edges_url)
        nodes_df = pd.read_csv(nodes_url)
        return edges_df, nodes_df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None, None

def create_graph(edges_df, nodes_df):
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        node_id = row.get('Id') or row.get('id')
        label = row.get('Label') or row.get('label') or str(node_id)
        G.add_node(node_id, label=label, title=label)
    for _, row in edges_df.iterrows():
        src = row.get('Source') or row.get('source')
        tgt = row.get('Target') or row.get('target')
        w = row.get('Weight') or row.get('weight') or 1
        if src in G.nodes and tgt in G.nodes:
            G.add_edge(src, tgt, weight=w)
    return G

def calculate_metrics(G):
    density = nx.density(G)
    degree_dict = nx.degree_centrality(G)
    betweenness_dict = nx.betweenness_centrality(G, weight='weight')
    partition = community_louvain.best_partition(G, weight='weight')
    modularity_score = community_louvain.modularity(partition, G)
    
    nx.set_node_attributes(G, degree_dict, 'degree_centrality')
    nx.set_node_attributes(G, betweenness_dict, 'betweenness_centrality')
    nx.set_node_attributes(G, partition, 'group')
    return G, density, modularity_score, degree_dict, betweenness_dict, partition

def visualize_network(G, partition):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    unique_communities = list(set(partition.values()))
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_map = {com: colors[i % len(colors)] for i, com in enumerate(unique_communities)}
    
    for node_id in G.nodes:
        node = G.nodes[node_id]
        size = node['degree_centrality'] * 30 + 10
        group_id = node['group']
        color = color_map.get(group_id, "#97C2FC")
        title_html = f"<b>{node['label']}</b><br>Degree: {node['degree_centrality']:.3f}<br>Betweenness: {node['betweenness_centrality']:.3f}<br>Group: {group_id}"
        net.add_node(node_id, label=node['label'], title=title_html, size=size, color=color, group=group_id)

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        net.add_edge(u, v, value=weight, color="#cccccc")

    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08,
        damping=0.4,
        overlap=0
    )
    
    try:
        path = "/tmp"
        net.save_graph(f'pyvis_graph.html')
        HtmlFile = open(f'pyvis_graph.html', 'r', encoding='utf-8')
        return HtmlFile.read()
    except:
        net.save_graph('pyvis_graph.html')
        HtmlFile = open('pyvis_graph.html', 'r', encoding='utf-8')
        return HtmlFile.read()

def main():
    st.sidebar.title("📖 导航栏")
    st.sidebar.info("李子睿 CHC5904 Hands-on Assignment2")
    
    selected_phase = st.sidebar.selectbox("选择分析阶段", list(DATA_SOURCES.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Research Question")
    st.sidebar.markdown("""
    **如何通过社会网络的变化体现贾宝玉的个人成长？**
    本研究通过对比三个关键人生阶段（天真、觉醒、冲击）的社交网络结构，
    分析贾宝玉在《红楼梦》大观园体系中的位置变迁。
    """)
    
    st.title(f"🕸️ {selected_phase}")
    st.markdown(f"_{DATA_SOURCES[selected_phase]['desc']}_")
    
    with st.spinner('正在从GitHub加载数据并运行算法...'):
        edges_url = DATA_SOURCES[selected_phase]['edges']
        nodes_url = DATA_SOURCES[selected_phase]['nodes']
        
        edges_df, nodes_df = load_data(edges_url, nodes_url)
        
        if edges_df is not None and nodes_df is not None:
            G = create_graph(edges_df, nodes_df)
            G, density, modularity, degree, betweenness, partition = calculate_metrics(G)
  
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

            st.subheader("🕸️ 交互式网络可视化 (Interactive Visualization)")
            st.markdown("节点大小 = 度中心性 | 节点颜色 = 社群 (Community) | 布局 = Force Atlas 2")
            html_data = visualize_network(G, partition)
            components.html(html_data, height=620)
 
            st.subheader("🔍 核心人物分析 (Centrality Analysis)")
            metrics_df = pd.DataFrame({
                'Character': [G.nodes[n]['label'] for n in G.nodes],
                'Degree (影响力)': [degree[n] for n in G.nodes],
                'Betweenness (桥接能力)': [betweenness[n] for n in G.nodes],
                'Community (社群)': [partition[n] for n in G.nodes]
            }).sort_values(by='Degree (影响力)', ascending=False)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**影响力排名 (Top by Degree)**")
                st.dataframe(metrics_df[['Character', 'Degree (影响力)']].head(10), use_container_width=True)
            with c2:
                st.markdown("**中介能力排名 (Top by Betweenness)**")
                st.dataframe(metrics_df.sort_values(by='Betweenness (桥接能力)', ascending=False)[['Character', 'Betweenness (桥接能力)']].head(10), use_container_width=True)
            
            st.markdown("---")
            st.subheader("📝 阶段性深度反思 (Phase Reflection)")
            
            def academic_insight(title, content, analysis):
                st.markdown(f"#### {title}")
                st.markdown(f"📖 **文本背景 (Context):** {content}")
                st.warning(f"💡 **SNA与Close Reading:** {analysis}")

            if "Phase 1" in selected_phase:
                academic_insight(
                    "1. 无差别的泛爱 (Universal Love)",
                    "在此阶段，大观园初建，第23回《西厢记妙词通戏语》是高潮。宝玉试图建立一个跨越阶级（丫鬟与小姐混同）的“女儿国”。",
                    "网络呈现**高密度 (High Density)** 与 **以自我为中心(Egocentric)** 的特征。通过社会网络分析，我们能清晰地看到贾宝玉身处一个关系紧密的小圈子里，所有人际关系都围绕着他展开。有趣的是，数据分析结果揭示了一个打破常规的现象，在贾宝玉的社交世界中，袭人、晴雯等丫鬟等节点，和林黛玉、薛宝钗等小姐的节点，与他的节点距离几乎是相等。在那个等级森严的时代，这一点显得尤为特别。这应征了贾宝玉的青春懵懂时期的“泛爱”特征，他对身边的女孩们保有一种超越阶级的深刻同情与体贴。从这个阶段贾宝玉的社会关系网可以看出，他试图在大观园的象牙塔内，抹平身份差异，构建一个属于他的青春乌托邦王国。"
                )
            elif "Phase 2" in selected_phase:
                academic_insight(
                    "2. 金木之争 (Differentiation & Tension)",
                    "第27回“滴翠亭”与第29回“清虚观打醮”加剧了“金玉良缘”与“木石前盟”的冲突。",
                    "网络的**Modularity (模块化)** 指标上升，显示社群开始分化。大观园的“理想国”开始出现裂痕。社会网络的分析显示出贾宝玉的核心社交圈产生了分化。他的主要情感链接向林黛玉倾斜，但薛宝钗凭借高超的社交智慧，连接不同群体，在社会关系网依然中有着重要的结构性地位。这个阶段的社会关系网络精准地捕捉到了，贾宝玉在“泛爱”到“情定”的转变中，所承受的结构性压力。他的大部分“情感带宽”都被这场三角纠葛所占据，导致他与网络中其他边缘成员的关系逐渐疏远，预示了未来更大范围的离散。"
                )
            elif "Phase 3" in selected_phase:
                academic_insight(
                    "3. 权力的入侵与桥梁的断裂 (Intrusion & Structural Collapse)",
                    "第33回宝玉挨打、第34回袭人进言。**贾政、王夫人** 等代表父权/封建秩序的节点权重急剧上升，直接冲击了大观园的内部网络。",
                    "**袭人** 的 **Betweenness Centrality (中介中心性)** 在此阶段极具研究意义。 她成为了连接两个世界的“枢纽”。一边是贾宝玉构建的青春乌托邦，另一边则是贾府森严的等级秩序。当王夫人的意志通过袭人这一环介入时，外部权力便强力“入侵”了宝玉的社交网络。社会网络分析清晰地显示，这种来自高层的强力干预，瞬间打破了园内自然形成的人际流动。宝玉不再是这个网络唯一的主宰，他所珍视的平等交往，正被无可抗拒的社会规则切割、重组。"
                )

        
            st.markdown("---")
            st.subheader("🎓 研究总结：被规训的乌托邦 (Research Conclusion)")
            st.markdown("""
            我们将这三个阶段的SNA分析融合来看，清晰地描绘出了**贾宝玉的成长悲剧**。
            他所谓的“成长”并非传统意义上的积极成熟，而是一个**理想的“乌托邦”被现实的“规则”慢慢攻入、解体并最终重塑的过程**。
            """)

            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.info("### 🏰 第一阶段：理想主义的搭建")
                st.markdown("**「我要缔造一个平等世界」**")
                st.caption("贾宝玉是一位散发着浪漫主义风采的“社会革命家”。他试图在小天地大观园里，搭建一个超出封建等级的新秩序。**高密度、去阶级属性的社交网络**，正是这一乌托邦理想的数据化呈现。他“成长”的发端，富有主动性与建构性的色彩。")

            with col_b:
                st.warning("### ⚡ 第二阶段：信念的动摇")
                st.markdown("**「我的生活世界开始出现裂痕」**")
                st.caption("当情感深度卷入（黛玉）与外部压力（金玉良缘）同时呈现，完美世界显露结构性矛盾。**网络模块化上升**，小群体组建。他不再是闲适的中心，而是被情感纠葛消耗了大量“社交容量”的纠结个体。成长表现为理想与现实冲突引发的痛苦。")
            
            with col_c:
                st.error("### ⛓️ 第三阶段：理想的消逝")
                st.markdown("**「我被现实约束，无力招架」**")
                st.caption("这是最残酷的阶段。封建父权（贾政/王夫人）借助守门人（袭人）强行闯入，**重写了社交网络规则**。宝玉在社交格局上丢掉了主导权，外力对网络进行“切割与重组”，说明他从组建者沦为被规训的对象。成长是带着痛苦的被动接纳。")

            st.markdown(
                """
                <div class="conclusion-box">
                    <b>结论 (Conclusion):</b><br>
                    这三部分的SNA分析共同展现出一个核心主题：<b>贾宝玉的“成长”，是个体理想不断被社会规则驯化的悲剧性过程。</b><br><br>
                    他起始是个试图抹除等级的叛逆者，经过内心的一番挣扎，最终成为被等级规则再次编排的牺牲品。<br>
                    相关的社交网络图谱直观呈现了他的青春乌托邦怎样从建立、产生裂缝到最后沦陷。<br>
                    这不只是属于贾宝玉个人的成长史，也是所有美好理想在无情现实面前怎样破碎的一个缩影。
                </div>
                """, 
                unsafe_allow_html=True
            )

        else:
            st.warning("无法加载数据，请检查CSV文件格式或网络连接。")

if __name__ == "__main__":
    main()
