"""
Phase 5 可视化: pyvis 交互式知识图谱
生成 HTML 文件, 可在浏览器中查看

用法:
  python code_p5_visualize.py
  python code_p5_visualize.py --input ./output/knowledge_graph.gpickle
  python code_p5_visualize.py --input ./output/knowledge_graph.json --format json
  python code_p5_visualize.py --max-nodes 200
"""
import argparse
import pickle
import json
import sys
from pathlib import Path
from collections import Counter

import networkx as nx
from loguru import logger


# ============================================================
# 节点颜色映射
# ============================================================

NODE_COLORS = {
    "person":  "#e74c3c",   # 红
    "project": "#3498db",   # 蓝
    "module":  "#1abc9c",   # 青
    "file":    "#9b59b6",   # 淡紫
    "bug":     "#f1c40f",   # 黄
    "tool":    "#e67e22",   # 桃
    "concept": "#2ecc71",   # 绿
    "unknown": "#95a5a6",   # 灰
}


def _patch_html_tooltip(html_path: str):
    """
    后处理 pyvis 生成的 HTML，注入自定义 tooltip 渲染代码

    vis-network 9.x 默认将 title 属性渲染为纯文本，
    此函数注入自定义 JavaScript 来正确渲染 HTML 内容
    """
    # 自定义 tooltip 的 JavaScript 代码
    tooltip_js = """
<script>
// 自定义 HTML tooltip 渲染
(function() {
    // 创建 tooltip 容器
    const tooltip = document.createElement('div');
    tooltip.id = 'custom-tooltip';
    tooltip.style.cssText = `
        position: absolute;
        display: none;
        background: rgba(30, 30, 50, 0.95);
        color: #fff;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 13px;
        line-height: 1.5;
        pointer-events: none;
        z-index: 9999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        max-width: 300px;
    `;
    document.body.appendChild(tooltip);

    // 等待 vis-network 初始化完成
    const checkNetwork = setInterval(() => {
        if (typeof network !== 'undefined' && network.body) {
            clearInterval(checkNetwork);
            initTooltip();
        }
    }, 100);

    function initTooltip() {
        // 隐藏原生 tooltip
        const style = document.createElement('style');
        style.textContent = '.vis-tooltip { display: none !important; }';
        document.head.appendChild(style);

        // 监听节点悬停事件
        network.on('hoverNode', function(params) {
            const node = network.body.data.nodes.get(params.node);
            if (node && node.title) {
                tooltip.innerHTML = node.title;
                tooltip.style.display = 'block';
            }
        });

        network.on('blurNode', function() {
            tooltip.style.display = 'none';
        });

        // 监听边悬停事件
        network.on('hoverEdge', function(params) {
            const edge = network.body.data.edges.get(params.edge);
            if (edge && edge.title) {
                tooltip.innerHTML = edge.title;
                tooltip.style.display = 'block';
            }
        });

        network.on('blurEdge', function() {
            tooltip.style.display = 'none';
        });

        // 跟随鼠标移动
        document.addEventListener('mousemove', function(e) {
            if (tooltip.style.display === 'block') {
                tooltip.style.left = (e.pageX + 15) + 'px';
                tooltip.style.top = (e.pageY + 15) + 'px';
            }
        });
    }
})();
</script>
"""

    # 读取 HTML 文件
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 在 </body> 前插入自定义 tooltip 代码
    if '</body>' in content:
        content = content.replace('</body>', f'{tooltip_js}\n</body>')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)


def visualize_graph(
    G: nx.MultiDiGraph,
    output_path: str = "./output/knowledge_graph.html",
    height: int = 800,
    width: int = 1200,
    physics: str = "forceAtlas2Based",
    max_nodes: int = 500,
    drop_isolated: bool = False,
):
    """
    用 pyvis 生成交互式知识图谱 HTML

    Args:
        G: NetworkX MultiDiGraph
        output_path: HTML 输出路径
        height: 画布高度 (px)
        width: 画布宽度 (px)
        physics: 物理引擎 (forceAtlas2Based / barnesHut / repulsion)
        max_nodes: 最大节点数 (超过则取 top-N 度节点)
        drop_isolated: 是否丢弃孤立节点 (degree=0)
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError(
            "pyvis 未安装, 请运行: pip install pyvis"
        )

    # 裁剪: 如果节点数超过 max_nodes, 取 top-N 度节点
    if G.number_of_nodes() > max_nodes:
        logger.warning(
            f"节点数 {G.number_of_nodes()} 超过 max_nodes={max_nodes}, "
            f"裁剪为 top-{max_nodes} 度节点"
        )
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = {n for n, _ in top_nodes}
        G = G.subgraph(top_node_names).copy()

    # 丢弃孤立节点
    if drop_isolated:
        isolates = list(nx.isolates(G))
        if isolates:
            G.remove_nodes_from(isolates)
            logger.info(f"丢弃 {len(isolates)} 个孤立节点")

    logger.info(f"可视化: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # 创建 pyvis Network
    # 使用 100vh/100% 实现满屏显示
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
    )

    # 设置物理引擎
    if physics == "forceAtlas2Based":
        net.force_atlas_2based(
            gravity=-50,
            central_gravity=0.01,
            spring_length=100,
            spring_strength=0.08,
            damping=0.4,
        )
    elif physics == "barnesHut":
        net.barnes_hut(
            gravity=-3000,
            central_gravity=0.3,
            spring_length=95,
            spring_strength=0.1,
            damping=0.09,
        )
    else:
        net.repulsion(
            node_distance=150,
            central_gravity=0.2,
            spring_length=150,
            spring_strength=0.05,
            damping=0.09,
        )

    # 启用 hover 交互 (hoverNode/hoverEdge 事件需要此选项)
    if hasattr(net, 'options') and hasattr(net.options, 'interaction'):
        net.options.interaction.hover = True
        net.options.interaction.tooltipDelay = 0
    elif isinstance(getattr(net, 'options', None), dict):
        net.options.setdefault('interaction', {})['hover'] = True
        net.options['interaction']['tooltipDelay'] = 0

    # 计算节点度 (用于缩放大小)
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_size = 10
    max_size = 50

    # 添加节点
    for node, data in G.nodes(data=True):
        degree = degrees.get(node, 0)
        # 大小按度线性缩放
        size = min_size + (max_size - min_size) * (degree / max_degree) if max_degree > 0 else min_size

        entity_type = data.get("entity_type", "unknown")
        color = NODE_COLORS.get(entity_type, NODE_COLORS["unknown"])

        # 构建 tooltip
        aliases = data.get("aliases", [])
        source_tasks = data.get("source_tasks", [])
        title_parts = [
            f"<b>{node}</b>",
            f"类型: {entity_type}",
            f"度: {degree}",
        ]
        if aliases:
            title_parts.append(f"别名: {', '.join(aliases[:5])}")
        if source_tasks:
            title_parts.append(f"来源 task: {len(source_tasks)} 个")
        title = "<br>".join(title_parts)

        # 标签: 如果名字太长则截断
        label = node if len(node) <= 20 else node[:18] + "..."

        net.add_node(
            node,
            label=label,
            title=title,
            size=size,
            color=color,
            font={"size": 12, "color": "white"},
        )

    # 添加边
    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "")
        weight = data.get("weight", 0.5)
        source_task = data.get("source_task", "")

        title = f"{relation}<br>置信度: {weight:.2f}"
        if source_task:
            title += f"<br>来源: {source_task[-12:]}"

        # 边宽按置信度缩放
        edge_width = 1 + weight * 3

        net.add_edge(
            u, v,
            title=title,
            label=relation,
            width=edge_width,
            font={"size": 10, "color": "#aaaaaa", "align": "middle"},
            color={"color": "#4a4a6a", "highlight": "#e74c3c"},
            arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
        )

    # 添加图例 (用隐藏节点实现)
    net.add_node(
        "__legend__",
        label="",
        title="",
        x=-width//2 + 100,
        y=-height//2 + 50,
        physics=False,
        hidden=True,
    )

    # 保存
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(p))

    # 后处理: 注入自定义 tooltip 渲染代码 (修复 HTML 标签不渲染的问题)
    _patch_html_tooltip(str(p))

    logger.info(f"可视化已保存: {output_path}")

    # 统计
    type_counts = Counter(
        data.get("entity_type", "unknown")
        for _, data in G.nodes(data=True)
    )
    logger.info(f"节点类型分布: {dict(type_counts)}")


def load_graph(input_path: str, fmt: str = "auto") -> nx.MultiDiGraph:
    """加载图谱文件"""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    if fmt == "auto":
        fmt = "json" if input_path.endswith(".json") else "gpickle"

    if fmt == "gpickle":
        with open(input_path, "rb") as f:
            return pickle.load(f)
    elif fmt == "json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return nx.node_link_graph(data, multigraph=True, directed=True, edges="links")
    else:
        raise ValueError(f"不支持的格式: {fmt}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5: 知识图谱可视化")
    parser.add_argument(
        "--input", default="./output/knowledge_graph.gpickle",
        help="图谱文件路径 (.gpickle 或 .json)",
    )
    parser.add_argument(
        "--format", default="auto",
        choices=["auto", "gpickle", "json"],
        help="文件格式",
    )
    parser.add_argument(
        "--output", default="./output/knowledge_graph.html",
        help="HTML 输出路径",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=500,
        help="最大节点数",
    )
    parser.add_argument(
        "--drop-isolated", action="store_true",
        help="丢弃孤立节点",
    )
    parser.add_argument(
        "--height", type=int, default=800,
        help="画布高度 (px)",
    )
    parser.add_argument(
        "--width", type=int, default=1200,
        help="画布宽度 (px)",
    )
    parser.add_argument(
        "--physics", default="forceAtlas2Based",
        choices=["forceAtlas2Based", "barnesHut", "repulsion"],
        help="物理引擎",
    )
    args = parser.parse_args()

    # 加载图谱
    logger.info(f"加载图谱: {args.input}")
    G = load_graph(args.input, args.format)
    logger.info(f"节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")

    # 可视化
    visualize_graph(
        G,
        output_path=args.output,
        height=args.height,
        width=args.width,
        physics=args.physics,
        max_nodes=args.max_nodes,
        drop_isolated=args.drop_isolated,
    )


if __name__ == "__main__":
    main()
