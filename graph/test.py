# -*- coding: utf-8 -*-
"""
生产级核心实现：Melanoma Knowledge Graph (Gremlin-Python)
==========================================================
设计目标
- 同步 Gremlin（无假 async）
- 只使用业务主键，不跨函数传 Vertex 对象
- 可重复执行（幂等）
- 查询无 N+1 问题
- 为 GraphRAG / Agent 预留结构

适用：JanusGraph / HugeGraph / Neptune / Gremlin Server
"""

from gremlin_python.driver.client import Client
from gremlin_python.process.traversal import T, Order
from typing import List, Dict
import time

# =========================
# 基础配置
# =========================

GREMLIN_URL = "ws://localhost:8090/gremlin"
GREMLIN_TRAVERSAL_SOURCE = "g"


# =========================
# 工具函数
# =========================

def today_ts():
    return int(time.time())


# =========================
# 核心系统
# =========================

class MelanomaGraphSystem:
    def __init__(self):
        self.client = Client(GREMLIN_URL, GREMLIN_TRAVERSAL_SOURCE)

    # ---------- 基础封装 ----------

    def submit(self, gremlin: str, bindings: Dict = None):
        return self.client.submit(gremlin, bindings or {}).all().result()

    def close(self):
        self.client.close()

    # ---------- 幂等创建 ----------

    def upsert_vertex(self, label: str, key: str, value, props: Dict):
        gremlin = f"""
        g.V().has('{label}','{key}',keyValue)
          .fold()
          .coalesce(unfold(),
                    addV('{label}').property('{key}', keyValue))
          {''.join([f".property('{k}', props['{k}'])" for k in props])}
        """
        bindings = {"keyValue": value, "props": props}
        self.submit(gremlin, bindings)

    def upsert_edge(self, from_label, from_key, from_value,
                    edge_label,
                    to_label, to_key, to_value):
        gremlin = f"""
        g.V().has('{from_label}','{from_key}',fromValue)
         .as('a')
         .V().has('{to_label}','{to_key}',toValue)
         .coalesce(
            inE('{edge_label}').where(outV().as('a')),
            addE('{edge_label}').from('a')
         )
        """
        self.submit(gremlin, {
            "fromValue": from_value,
            "toValue": to_value
        })

    # =========================
    # Schema（概念层）
    # =========================

    def create_schema(self):
        concepts = [
            ("Concept", "患者"), ("Concept", "医生"), ("Concept", "样本"),
            ("Concept", "基因突变"), ("Concept", "药物"), ("Concept", "随访")
        ]
        for _, name in concepts:
            self.upsert_vertex("Concept", "name", name, {
                "description": name
            })

    # =========================
    # 实体创建
    # =========================

    def create_doctor(self, doctor_id: str, name: str):
        self.upsert_vertex(
            "医生", "医生ID", doctor_id,
            {"姓名": name}
        )

    def create_patient(self, patient_id: str, gender: str, age: int):
        self.upsert_vertex(
            "患者", "患者ID", patient_id,
            {"性别": gender, "年龄": age}
        )

    def bind_doctor_patient(self, doctor_id: str, patient_id: str):
        self.upsert_edge(
            "医生", "医生ID", doctor_id,
            "治疗",
            "患者", "患者ID", patient_id
        )

    def create_sample(self, sample_id: str, patient_id: str):
        self.upsert_vertex(
            "样本", "样本ID", sample_id,
            {}
        )
        self.upsert_edge(
            "患者", "患者ID", patient_id,
            "拥有",
            "样本", "样本ID", sample_id
        )

    def create_mutation(self, gene: str, sample_id: str):
        self.upsert_vertex(
            "基因突变", "基因名", gene,
            {}
        )
        self.upsert_edge(
            "样本", "样本ID", sample_id,
            "包含",
            "基因突变", "基因名", gene
        )

    def create_drug(self, drug_name: str):
        self.upsert_vertex(
            "药物", "药物名称", drug_name,
            {}
        )

    def prescribe_drug(self, patient_id: str, drug_name: str):
        self.upsert_edge(
            "患者", "患者ID", patient_id,
            "用药",
            "药物", "药物名称", drug_name
        )

    def create_followup(self, followup_id: str, patient_id: str,
                        date_str: str, status: str, recurrence: str):
        self.upsert_vertex(
            "随访", "随访ID", followup_id,
            {
                "随访日期": date_str,
                "随访时间戳": today_ts(),
                "生存状态": status,
                "是否复发": recurrence
            }
        )
        self.upsert_edge(
            "患者", "患者ID", patient_id,
            "记录",
            "随访", "随访ID", followup_id
        )

    # =========================
    # 查询（生产可用）
    # =========================

    def patients_with_braf_nras(self) -> List[str]:
        # 一、先给你一句“总感觉”
        # Gremlin 查询 = 人在图里“走路 + 做事”
        # 你可以把 Gremlin 想象成：
        #   “我站在图里的某些点上，然后一步一步往外走，每走一步就筛选、加工、记录”
        # 二、最核心的几个“起点”和“动作”
        # g “给我这张图” == SQL 里的 FROM database
        # g.V() “我站在所有节点上”(所有患者 + 医生 + 样本 + 药物 + 随访 ……)
        # g.V().hasLabel("患者") == WHERE label = '患者' “在所有节点里，只看【患者】”
        # g.V().has("患者", "患者ID", "P001") == “找标签是【患者】，而且患者ID 是 P001 的那个人”
        # 三、最重要的：“走路”相关的关键词
        # g.V().hasLabel("患者").out("拥有") ==  “从患者出发，顺着【拥有】这条关系，走到另一端”  患者 --拥有--> 样本 “找到患者的样本”
        # g.V().hasLabel("样本").in("拥有") == “从样本，反过来找到拥有它的患者”
        # both("治疗") == “不管箭头方向，只要和【治疗】有关就走”
        # 四、筛选 & 过滤（像查名单）
        # .where(values("年龄").is(gt(50))) ==  “只留下年龄大于 50 的”
        # dedup() 去重
        # 五、拿结果“做点事”
        # values("字段") —— 只拿某个属性 => values("患者ID") => “别给我整个对象，只给我患者ID”
        # valueMap() —— 把属性打包给我  => valueMap(true) => “把这个节点的所有信息一次性给我”
        # count() —— 数一数 =>  count() =>  “有多少个？”
        # 六、排序 & 统计（你以后会常用）
        # order().by() —— 排序 => order().by("随访时间戳", desc) => “按随访时间，从近到远排”
        # group() —— 分组统计 => group().by("生存状态").by(count()) => “按生存状态分组，每组数一数有多少人”
        # 七、临时起名字（这是高手和新手的分水岭）
        # as("名字") —— 给当前位置起个外号 => as("患者") => “我现在站的这个位置，叫它【患者】”
        # select("名字") —— 回到那个位置 => select("患者") => “回到刚才那个患者”
        # 例子（非常重要）
        #   g.V().hasLabel("患者"). as ("p")
        #   .out("拥有").out("包含").has("基因名", "BRAF")
        #   .select("p")
        # 人话：
        #   找患者 → 记住他 →
        #   看他的样本 → 找到BRAF突变 →
        #   回到这个患者本身
        # 八、最终：读一整句 Gremlin（像读中文）
        # g.V().hasLabel("患者").as("p")
        #  .out("拥有").out("包含").has("基因名","BRAF")
        #  .select("p").values("患者ID").dedup()
        # 用人话翻译
        #   从所有患者出发 →
        #   记住患者本人 →
        #   看他有没有样本 →
        #   样本里有没有 BRAF 突变 →
        #   如果有，就回到这个患者 →
        #   输出他的患者ID →
        #   去重
        # 九、给你一个“初学者心法”（非常重要）
        # 🔑 Gremlin 学习三问法
        # 每一行你都问自己三句话：
        # 1️⃣ 我现在站在哪？
        # 2️⃣ 我往哪走？
        # 3️⃣ 我要筛掉谁？留下谁？
        # 只要你能回答这三句，你就会写 Gremlin。
        # 🔚 最后一句实话
        # Gremlin 不是“查询语言”，
        # 它是“在图里走路的语言”
        gremlin = """
        g.V().hasLabel('患者').as('p')
         .out('拥有').as('s')
         .out('包含').has('基因名','BRAF').select('s')
         .out('包含').has('基因名','NRAS')
         .select('p').values('患者ID').dedup()
        """
        return self.submit(gremlin)

    def doctor_workload(self):
        gremlin = """
        g.V().hasLabel('医生').project('医生','患者数')
         .by('姓名')
         .by(out('治疗').dedup().count())
        """
        return self.submit(gremlin)

    def latest_followup(self, patient_id: str):
        gremlin = """
        g.V().has('患者','患者ID',pid)
         .out('记录')
         .order().by('随访时间戳', desc)
         .limit(1)
         .valueMap(true)
        """
        return self.submit(gremlin, {"pid": patient_id})


# =========================
# Demo 主流程
# =========================

if __name__ == '__main__':
    system = MelanomaGraphSystem()

    try:
        system.create_schema()

        system.create_doctor("D001", "张医生")
        system.create_patient("P001", "男", 55)
        system.bind_doctor_patient("D001", "P001")

        system.create_sample("S001", "P001")
        system.create_mutation("BRAF", "S001")
        system.create_mutation("NRAS", "S001")

        system.create_drug("维莫非尼")
        system.prescribe_drug("P001", "维莫非尼")

        system.create_followup("FU001", "P001", "2024-01-01", "存活", "否")

        print("BRAF+NRAS 患者:", system.patients_with_braf_nras())
        print("医生工作负荷:", system.doctor_workload())

    finally:
        system.close()
