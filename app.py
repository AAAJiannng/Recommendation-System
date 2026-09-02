import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("游戏.csv")
    
    user_game_matrix = df.pivot_table(
        index='用户ID', columns='游戏', values='游戏时间', aggfunc='sum'
    ).fillna(0)
    
    user_game_matrix_binary = user_game_matrix.applymap(lambda x: 1 if x > 0 else 0)
    
    user_sim = pd.DataFrame(
        cosine_similarity(user_game_matrix_binary),
        index=user_game_matrix_binary.index,
        columns=user_game_matrix_binary.index
    )
    
    item_sim = pd.DataFrame(
        cosine_similarity(user_game_matrix_binary.T),
        index=user_game_matrix_binary.columns,
        columns=user_game_matrix_binary.columns
    )
    
    return df, user_game_matrix_binary, user_sim, item_sim

df, user_game_matrix, user_sim, item_sim = load_and_preprocess()

def recommend_by_user(user_id, top_n=10):
    """基于用户的协同过滤：找到与用户最相似的Top K用户，推荐他们玩过而该用户未玩的游戏"""
    if user_id not in user_sim.index:
        return None
    
    sim_series = user_sim.loc[user_id].sort_values(ascending=False)
    sim_series = sim_series[sim_series.index != user_id]
    
    top_users = sim_series.head(5).index
    
    recommended = set()
    for u in top_users:
        games_u = set(user_game_matrix.loc[u][user_game_matrix.loc[u] == 1].index)
        games_user = set(user_game_matrix.loc[user_id][user_game_matrix.loc[user_id] == 1].index)
        recommended.update(games_u - games_user)
    
    return list(recommended)[:top_n]

def recommend_by_item(item_name, top_n=10):
    """基于物品的协同过滤：找出与给定游戏最相似的Top N游戏"""
    if item_name not in item_sim.columns:
        return None
    
    sim_series = item_sim[item_name].sort_values(ascending=False)
    sim_series = sim_series[sim_series.index != item_name]
    return sim_series.head(top_n).index.tolist()

st.set_page_config(page_title="推荐系统在线看板", layout="centered")
st.title("🎮推荐系统实战 - 基于协同过滤")
st.markdown("### 数据概况")
st.write(f"总用户数：{len(user_game_matrix)}，总游戏数：{len(user_game_matrix.columns)}")

mode = st.radio("选择推荐模式", ("基于用户 (User-based)", "基于物品 (Item-based)"))

if mode == "基于用户 (User-based)":
    user_id_input = st.text_input("请输入用户ID (例如 86540)")
    if st.button("推荐游戏"):
        if user_id_input.isdigit():
            user_id = int(user_id_input)
            recs = recommend_by_user(user_id)
            if recs is None:
                st.error("用户ID不存在，请重新输入")
            else:
                st.success(f"为用户 {user_id} 推荐的游戏：")
                for i, game in enumerate(recs, 1):
                    st.write(f"{i}. {game}")
        else:
            st.error("请输入有效的数字用户ID")

else:  
    item_name_input = st.text_input("请输入游戏名称 (例如 游戏163)")
    if st.button("查找相似游戏"):
        recs = recommend_by_item(item_name_input)
        if recs is None:
            st.error("游戏名称不存在，请重新输入")
        else:
            st.success(f"与 {item_name_input} 相似的游戏：")
            for i, game in enumerate(recs, 1):
                st.write(f"{i}. {game}")

with st.expander("查看相似度矩阵样例（前5行）"):
    st.write("用户相似度矩阵 (前5个用户)：")
    st.dataframe(user_sim.iloc[:5, :5])
    st.write("物品相似度矩阵 (前5个游戏)：")
    st.dataframe(item_sim.iloc[:5, :5])

st.caption("提示：首次加载需计算相似度矩阵，请耐心等待。数据来源于 '游戏.csv'")