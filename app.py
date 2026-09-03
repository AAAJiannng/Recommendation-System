import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("游戏.csv")

    matrix = df.pivot_table(
        index='用户ID', columns='游戏', values='游戏时间', aggfunc='sum'
    ).fillna(0)
    matrix_binary = matrix.applymap(lambda x: 1 if x > 0 else 0)
    
    return df, matrix_binary

df, matrix = load_and_preprocess()

@st.cache_resource
def fit_user_knn(data, n_neighbors=6):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(data)
    return model

@st.cache_resource
def fit_item_knn(data, n_neighbors=11):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(data.T)
    return model

user_knn = fit_user_knn(matrix.values)
item_knn = fit_item_knn(matrix.values)

def recommend_by_user(user_id, top_n=10):
    """
    基于用户的协同过滤：
    找到与目标用户最相似的K个用户，推荐他们玩过而目标用户未玩的游戏。
    按共同出现次数排序，返回前top_n个。
    """
    if user_id not in matrix.index:
        return None
    
    idx = matrix.index.get_loc(user_id)
    distances, indices = user_knn.kneighbors(
        matrix.iloc[[idx]].values, n_neighbors=6
    )
    neighbor_indices = indices[0][1:]
    neighbor_ids = matrix.index[neighbor_indices]
    
    user_games = set(matrix.loc[user_id][matrix.loc[user_id] == 1].index)
    
    rec_counter = {}
    for uid in neighbor_ids:
        games = set(matrix.loc[uid][matrix.loc[uid] == 1].index)
        for g in games - user_games:
            rec_counter[g] = rec_counter.get(g, 0) + 1
    
    sorted_recs = sorted(rec_counter.items(), key=lambda x: x[1], reverse=True)
    return [game for game, _ in sorted_recs[:top_n]]


def recommend_by_item(item_name, top_n=10):
    """
    基于物品的协同过滤：
    找出与输入游戏最相似的K个游戏（排除自身）。
    """
    if item_name not in matrix.columns:
        return None
    
    idx = matrix.columns.get_loc(item_name)
    distances, indices = item_knn.kneighbors(
        matrix.T.iloc[[idx]].values, n_neighbors=11
    )
    neighbor_indices = indices[0][1:]
    neighbor_items = matrix.columns[neighbor_indices]
    return neighbor_items.tolist()[:top_n]


st.set_page_config(page_title="推荐系统在线看板", layout="centered")
st.title("🎮推荐系统实战-KNN协同过滤")
st.markdown(f"**数据概况**：共{len(matrix)}个用户，{len(matrix.columns)}款游戏")

mode = st.radio("选择推荐模式", ("基于用户 (User-based)", "基于物品 (Item-based)"))

if mode == "基于用户 (User-based)":
    user_input = st.text_input("请输入用户ID (例如86540)")
    if st.button("为我推荐游戏"):
        if not user_input.isdigit():
            st.error("请输入有效的数字用户ID")
        else:
            user_id = int(user_input)
            recs = recommend_by_user(user_id)
            if recs is None:
                st.error("用户ID不存在，请重新输入")
            elif len(recs) == 0:
                st.info("没有找到合适的推荐，可能是该用户已玩过所有邻居的游戏。")
            else:
                st.success(f"为您推荐的{len(recs)}款游戏：")
                for i, game in enumerate(recs, 1):
                    st.write(f"{i}. {game}")

else:  # 基于物品
    item_input = st.text_input("请输入游戏名称 (例如游戏163)")
    if st.button("查找相似游戏"):
        recs = recommend_by_item(item_input)
        if recs is None:
            st.error("游戏名称不存在，请重新输入")
        else:
            st.success(f"与{item_input}最相似的{len(recs)}款游戏：")
            for i, game in enumerate(recs, 1):
                st.write(f"{i}. {game}")

with st.expander("🔍查看交互矩阵样例（前5行）"):
    st.dataframe(matrix.iloc[:5, :5])

st.caption("💡提示：首次加载需训练KNN模型（约几秒），之后会缓存。若数据量极大，可修改数据加载部分的降维参数。")
