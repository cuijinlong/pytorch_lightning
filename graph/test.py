# -*- coding: utf-8 -*-
"""
医疗知识图谱系统 - 增强版
==========================================================
功能特性：
1. 完整的医院-科室-医生层级结构
2. 丰富的边属性（治疗、用药、会诊等）
3. 完整的g.E()按边查询示例
4. 幂等数据创建
5. 生产级查询封装
"""

from gremlin_python.driver.client import Client
from gremlin_python.process.traversal import T, Order, Direction
from typing import List, Dict, Optional
import time
from datetime import datetime

# =========================
# 基础配置
# =========================

GREMLIN_URL = "ws://localhost:8090/gremlin"
GREMLIN_TRAVERSAL_SOURCE = "g"


# =========================
# 工具函数
# =========================

def today_ts():
    """获取当前时间戳"""
    return int(time.time())


def format_date(year, month, day):
    """格式化日期"""
    return f"{year}-{month:02d}-{day:02d}"


def get_current_date():
    """获取当前日期字符串"""
    return datetime.now().strftime("%Y-%m-%d")


# =========================
# 核心系统类
# =========================

class MelanomaGraphSystem:
    def __init__(self):
        self.client = Client(GREMLIN_URL, GREMLIN_TRAVERSAL_SOURCE)
        self.edge_counter = 0  # 用于生成唯一的边ID

    # ---------- 基础封装 ----------

    def submit(self, gremlin: str, bindings: Dict = None):
        """提交Gremlin查询"""
        try:
            result = self.client.submit(gremlin, bindings or {}).all().result()
            return result
        except Exception as e:
            print(f"查询失败: {e}")
            print(f"查询语句: {gremlin}")
            return []

    def close(self):
        """关闭连接"""
        self.client.close()

    # ---------- 顶点操作 ----------

    def upsert_vertex(self, label: str, key: str, value, props: Dict):
        """创建或更新顶点（幂等）"""
        gremlin = f"""
        g.V().has('{label}','{key}',keyValue)
          .fold()
          .coalesce(unfold(),
                    addV('{label}').property('{key}', keyValue))
          {''.join([f".property('{k}', props['{k}'])" for k in props])}
        """
        bindings = {"keyValue": value, "props": props}
        self.submit(gremlin, bindings)

    # ---------- 边操作 ----------

    def upsert_edge(self, from_label, from_key, from_value,
                    edge_label,
                    to_label, to_key, to_value):
        """创建或更新边（不带属性）"""
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

    def upsert_edge_with_props(self, from_label, from_key, from_value,
                               edge_label,
                               to_label, to_key, to_value,
                               edge_props: Dict = None,
                               edge_id: str = None):
        """创建或更新边（带属性）"""
        # 生成边ID
        if edge_id is None:
            self.edge_counter += 1
            edge_id = f"E{self.edge_counter:06d}"

        # 先删除可能存在的重复边，然后创建带属性的边
        gremlin = f"""
        // 删除可能存在的重复边
        g.V().has('{from_label}','{from_key}',fromValue)
         .outE('{edge_label}')
         .where(inV().has('{to_label}','{to_key}',toValue))
         .drop()

        // 创建新边
        g.V().has('{from_label}','{from_key}',fromValue)
         .as('a')
         .V().has('{to_label}','{to_key}',toValue)
         .addE('{edge_label}').from('a')
         .property('edge_id', '{edge_id}')
         {''.join([f".property('{k}', edgeProps['{k}'])" for k in (edge_props or {})])}
        """
        bindings = {
            "fromValue": from_value,
            "toValue": to_value,
            "edgeProps": edge_props or {}
        }
        self.submit(gremlin, bindings)
        return edge_id

    def update_edge_property(self, edge_id: str, key: str, value):
        """更新边属性"""
        gremlin = f"""
        g.E().has('edge_id', '{edge_id}')
         .property('{key}', value)
        """
        self.submit(gremlin, {"value": value})

    def delete_edge(self, edge_id: str):
        """删除边"""
        gremlin = f"""
        g.E().has('edge_id', '{edge_id}').drop()
        """
        self.submit(gremlin)

    # =========================
    # Schema（概念层）
    # =========================

    def create_schema(self):
        """创建概念层"""
        concepts = [
            ("Concept", "患者"), ("Concept", "医生"), ("Concept", "样本"),
            ("Concept", "基因突变"), ("Concept", "药物"), ("Concept", "随访"),
            ("Concept", "科室"), ("Concept", "医院"), ("Concept", "治疗方案"),
            ("Concept", "临床分期"), ("Concept", "病理报告"), ("Concept", "处方"),
            ("Concept", "检查报告"), ("Concept", "手术记录")
        ]
        for _, name in concepts:
            self.upsert_vertex("Concept", "name", name, {
                "description": name
            })

    # =========================
    # 医院、科室层级结构
    # =========================

    def create_hospital(self, hospital_id: str, name: str, level: str,
                        city: str, address: str, phone: str):
        """创建医院"""
        self.upsert_vertex(
            "医院", "医院ID", hospital_id,
            {
                "医院名称": name,
                "医院等级": level,
                "所在城市": city,
                "地址": address,
                "联系电话": phone,
                "创建时间": today_ts(),
                "状态": "运营中"
            }
        )

    def create_department(self, dept_id: str, name: str, dept_type: str,
                          director: str, phone: str, bed_count: int = 0):
        """创建科室"""
        self.upsert_vertex(
            "科室", "科室ID", dept_id,
            {
                "科室名称": name,
                "科室类型": dept_type,
                "科室主任": director,
                "科室电话": phone,
                "床位数": bed_count,
                "创建时间": today_ts(),
                "状态": "正常"
            }
        )

    def bind_hospital_department(self, hospital_id: str, dept_id: str,
                                 relation_type: str = "包含"):
        """医院包含科室"""
        self.upsert_edge_with_props(
            "医院", "医院ID", hospital_id,
            relation_type,
            "科室", "科室ID", dept_id,
            {
                "关系类型": "组织结构",
                "创建时间": today_ts(),
                "状态": "有效"
            }
        )

    # =========================
    # 医生管理
    # =========================

    def create_doctor(self, doctor_id: str, name: str,
                      gender: str, age: int, title: str,
                      specialty: str, years_exp: int,
                      dept_id: str, phone: str, email: str):
        """创建医生并关联到科室"""
        self.upsert_vertex(
            "医生", "医生ID", doctor_id,
            {
                "姓名": name,
                "性别": gender,
                "年龄": age,
                "职称": title,
                "专业特长": specialty,
                "从业年限": years_exp,
                "手机号": phone,
                "邮箱": email,
                "工号": f"DOC{doctor_id}",
                "入职时间": today_ts(),
                "状态": "在职"
            }
        )

        # 关联到科室
        self.upsert_edge_with_props(
            "医生", "医生ID", doctor_id,
            "属于",
            "科室", "科室ID", dept_id,
            {
                "入职日期": get_current_date(),
                "职位": title,
                "创建时间": today_ts()
            }
        )

    # =========================
    # 患者管理
    # =========================

    def create_patient(self, patient_id: str, name: str,
                       gender: str, age: int, id_card: str,
                       phone: str, address: str, blood_type: str,
                       admission_date: str):
        """创建患者"""
        self.upsert_vertex(
            "患者", "患者ID", patient_id,
            {
                "姓名": name,
                "性别": gender,
                "年龄": age,
                "身份证号": id_card,
                "联系电话": phone,
                "住址": address,
                "血型": blood_type,
                "入院日期": admission_date,
                "创建时间": today_ts(),
                "状态": "在院"
            }
        )

    # =========================
    # 治疗关系（带丰富属性）
    # =========================

    def create_treatment_relationship(self, doctor_id: str, patient_id: str,
                                      treatment_type: str, start_date: str,
                                      end_date: str = None, status: str = "进行中",
                                      cost: float = 0.0, insurance_rate: float = 0.0,
                                      notes: str = ""):
        """创建治疗关系"""
        edge_id = self.upsert_edge_with_props(
            "医生", "医生ID", doctor_id,
            "治疗",
            "患者", "患者ID", patient_id,
            {
                "治疗类型": treatment_type,
                "开始时间": start_date,
                "结束时间": end_date or "",
                "治疗状态": status,
                "治疗费用": float(cost),
                "医保报销比例": float(insurance_rate),
                "备注": notes,
                "创建时间": today_ts(),
                "最后更新时间": today_ts()
            }
        )
        return edge_id

    # =========================
    # 药品与处方管理
    # =========================

    def create_drug(self, drug_name: str, drug_type: str,
                    manufacturer: str, indications: str, price: float):
        """创建药品"""
        self.upsert_vertex(
            "药物", "药物ID", f"DRUG-{drug_name}",
            {
                "药物名称": drug_name,
                "药物类型": drug_type,
                "生产厂家": manufacturer,
                "适应症": indications,
                "单价": float(price),
                "医保类别": "乙类",
                "创建时间": today_ts()
            }
        )

    def create_prescription(self, prescription_id: str, doctor_id: str,
                            patient_id: str, issue_date: str, status: str = "待取药"):
        """创建处方"""
        self.upsert_vertex(
            "处方", "处方ID", prescription_id,
            {
                "开方医生ID": doctor_id,
                "患者ID": patient_id,
                "开方日期": issue_date,
                "处方状态": status,
                "创建时间": today_ts()
            }
        )

        # 关联医生-处方
        self.upsert_edge_with_props(
            "医生", "医生ID", doctor_id,
            "开具",
            "处方", "处方ID", prescription_id,
            {
                "开方时间": issue_date,
                "开方类型": "门诊处方",
                "创建时间": today_ts()
            }
        )

        # 关联处方-患者
        self.upsert_edge_with_props(
            "处方", "处方ID", prescription_id,
            "对应",
            "患者", "患者ID", patient_id,
            {
                "关系类型": "用药关系",
                "创建时间": today_ts()
            }
        )

        return prescription_id

    def add_drug_to_prescription(self, prescription_id: str, drug_name: str,
                                 dosage: str, frequency: str, duration: str,
                                 quantity: int = 1):
        """处方中添加药品"""
        edge_id = self.upsert_edge_with_props(
            "处方", "处方ID", prescription_id,
            "包含药品",
            "药物", "药物ID", f"DRUG-{drug_name}",
            {
                "用法用量": dosage,
                "用药频次": frequency,
                "用药时长": duration,
                "数量": quantity,
                "创建时间": today_ts()
            }
        )
        return edge_id

    # =========================
    # 会诊关系
    # =========================

    def create_consultation(self, doctor_id: str, patient_id: str,
                            consultation_type: str, consultation_date: str,
                            duration: int, participants: List[str],
                            conclusion: str, cost: float = 0.0):
        """创建会诊关系"""
        edge_id = self.upsert_edge_with_props(
            "医生", "医生ID", doctor_id,
            "会诊",
            "患者", "患者ID", patient_id,
            {
                "会诊类型": consultation_type,
                "会诊时间": consultation_date,
                "会诊时长": duration,
                "参与人员": ",".join(participants),
                "会诊结论": conclusion,
                "会诊费用": float(cost),
                "创建时间": today_ts()
            }
        )
        return edge_id

    # =========================
    # 手术关系
    # =========================

    def create_surgery_relationship(self, doctor_id: str, patient_id: str,
                                    surgery_name: str, surgery_date: str,
                                    duration: int, anesthesia_type: str,
                                    success: bool = True, cost: float = 0.0):
        """创建手术关系"""
        edge_id = self.upsert_edge_with_props(
            "医生", "医生ID", doctor_id,
            "手术",
            "患者", "患者ID", patient_id,
            {
                "手术名称": surgery_name,
                "手术时间": surgery_date,
                "手术时长": duration,
                "麻醉方式": anesthesia_type,
                "手术成功": success,
                "手术费用": float(cost),
                "创建时间": today_ts()
            }
        )
        return edge_id

    # =========================
    # 医嘱关系
    # =========================

    def create_medical_order(self, doctor_id: str, patient_id: str,
                             order_type: str, order_content: str,
                             order_date: str, nurse: str = "",
                             status: str = "待执行"):
        """创建医嘱关系"""
        edge_id = self.upsert_edge_with_props(
            "医生", "医生ID", doctor_id,
            "医嘱",
            "患者", "患者ID", patient_id,
            {
                "医嘱类型": order_type,
                "医嘱内容": order_content,
                "开嘱时间": order_date,
                "执行护士": nurse,
                "执行状态": status,
                "创建时间": today_ts()
            }
        )
        return edge_id

    # =========================
    # 医患沟通关系
    # =========================

    def create_communication(self, doctor_id: str, patient_id: str,
                             communication_type: str, communication_date: str,
                             duration: int, topic: str, patient_feedback: str,
                             satisfaction: int = 5):
        """创建医患沟通关系"""
        edge_id = self.upsert_edge_with_props(
            "医生", "医生ID", doctor_id,
            "沟通",
            "患者", "患者ID", patient_id,
            {
                "沟通方式": communication_type,
                "沟通时间": communication_date,
                "沟通时长": duration,
                "沟通主题": topic,
                "患者反馈": patient_feedback,
                "满意度评分": satisfaction,
                "创建时间": today_ts()
            }
        )
        return edge_id

    # =========================
    # 转诊关系
    # =========================

    def create_referral(self, from_doctor_id: str, to_doctor_id: str,
                        patient_id: str, reason: str, referral_date: str,
                        referral_type: str = "科间转诊"):
        """创建转诊关系"""
        edge_id = self.upsert_edge_with_props(
            "医生", "医生ID", from_doctor_id,
            "转诊",
            "医生", "医生ID", to_doctor_id,
            {
                "转诊患者ID": patient_id,
                "转诊原因": reason,
                "转诊时间": referral_date,
                "转诊类型": referral_type,
                "创建时间": today_ts()
            }
        )
        return edge_id

    # =========================
    # 同事关系
    # =========================

    def create_colleague_relationship(self, doctor1_id: str, doctor2_id: str,
                                      cooperation_times: int = 1,
                                      last_cooperation: str = "",
                                      relationship: str = "同事"):
        """创建同事关系"""
        edge_id = self.upsert_edge_with_props(
            "医生", "医生ID", doctor1_id,
            "同事",
            "医生", "医生ID", doctor2_id,
            {
                "合作次数": cooperation_times,
                "最近合作": last_cooperation or get_current_date(),
                "合作关系": relationship,
                "创建时间": today_ts()
            }
        )
        return edge_id

    # =========================
    # 样本与基因突变
    # =========================

    def create_sample(self, sample_id: str, patient_id: str,
                      sample_type: str, collection_date: str):
        """创建样本"""
        self.upsert_vertex(
            "样本", "样本ID", sample_id,
            {
                "样本类型": sample_type,
                "采集日期": collection_date,
                "状态": "已接收",
                "创建时间": today_ts()
            }
        )

        # 关联患者-样本
        self.upsert_edge_with_props(
            "患者", "患者ID", patient_id,
            "拥有",
            "样本", "样本ID", sample_id,
            {
                "采集时间": collection_date,
                "采集人员": "",
                "创建时间": today_ts()
            }
        )

    def create_gene_mutation(self, gene: str, sample_id: str,
                             mutation_type: str, allele_frequency: float):
        """创建基因突变"""
        mutation_id = f"{gene}-{sample_id}-{today_ts()}"
        self.upsert_vertex(
            "基因突变", "突变ID", mutation_id,
            {
                "基因名": gene,
                "突变类型": mutation_type,
                "等位基因频率": float(allele_frequency),
                "检测方法": "NGS",
                "检测时间": get_current_date(),
                "临床意义": "",
                "创建时间": today_ts()
            }
        )

        # 关联样本-基因突变
        self.upsert_edge_with_props(
            "样本", "样本ID", sample_id,
            "包含",
            "基因突变", "突变ID", mutation_id,
            {
                "检测结果": "阳性",
                "创建时间": today_ts()
            }
        )

        return mutation_id

    # =========================
    # 随访记录
    # =========================

    def create_followup(self, followup_id: str, patient_id: str,
                        doctor_id: str, followup_date: str,
                        status: str, recurrence: str, notes: str):
        """创建随访记录"""
        self.upsert_vertex(
            "随访", "随访ID", followup_id,
            {
                "随访日期": followup_date,
                "随访时间戳": today_ts(),
                "生存状态": status,
                "是否复发": recurrence,
                "随访记录": notes,
                "随访医生ID": doctor_id,
                "创建时间": today_ts()
            }
        )

        # 关联患者-随访
        self.upsert_edge_with_props(
            "患者", "患者ID", patient_id,
            "记录",
            "随访", "随访ID", followup_id,
            {
                "记录类型": "随访",
                "创建时间": today_ts()
            }
        )

        # 关联医生-随访
        self.upsert_edge_with_props(
            "医生", "医生ID", doctor_id,
            "执行随访",
            "随访", "随访ID", followup_id,
            {
                "随访方式": "门诊",
                "创建时间": today_ts()
            }
        )

    # =========================
    # 查询功能
    # =========================

    def get_all_edges(self, limit: int = 100):
        """查询所有边"""
        gremlin = f"""
        g.E()
         .limit({limit})
         .elementMap()
        """
        return self.submit(gremlin)

    def get_edges_by_label(self, label: str, limit: int = 50):
        """按标签查询边"""
        gremlin = f"""
        g.E().hasLabel('{label}')
         .limit({limit})
         .elementMap()
        """
        return self.submit(gremlin)

    def get_edges_with_property(self, label: str, key: str, value):
        """查询具有特定属性的边"""
        gremlin = f"""
        g.E().hasLabel('{label}')
         .has('{key}', '{value}')
         .elementMap()
        """
        return self.submit(gremlin)

    def get_edges_between_vertices(self, from_id: str, to_id: str, label: str = None):
        """查询两个顶点之间的边"""
        if label:
            gremlin = f"""
            g.V('{from_id}').outE('{label}')
             .where(inV().hasId('{to_id}'))
             .elementMap()
            """
        else:
            gremlin = f"""
            g.V('{from_id}').bothE()
             .where(otherV().hasId('{to_id}'))
             .elementMap()
            """
        return self.submit(gremlin)

    def get_edge_statistics(self):
        """获取边统计信息"""
        gremlin = """
        g.E()
         .project('边标签', '起点标签', '终点标签')
         .by(label)
         .by(outV().label())
         .by(inV().label())
         .group()
           .by(
             project('起点标签', '边标签', '终点标签')
               .by('起点标签')
               .by('边标签')
               .by('终点标签')
           )
           .by(count())
         .unfold()
         .order().by(select(values), desc)
        """
        return self.submit(gremlin)

    def find_edges_by_time_range(self, label: str, time_key: str,
                                 start_time: str, end_time: str):
        """按时间范围查询边"""
        gremlin = f"""
        g.E().hasLabel('{label}')
         .has('{time_key}', between('{start_time}', '{end_time}'))
         .project('边ID', '时间', '起点', '终点', '属性')
         .by(id())
         .by('{time_key}')
         .by(outV().valueMap('姓名', '医生ID'))
         .by(inV().valueMap('姓名', '患者ID'))
         .by(valueMap())
         .order().by('{time_key}', asc)
        """
        return self.submit(gremlin)

    def get_edge_properties_summary(self, label: str):
        """获取边属性摘要"""
        gremlin = f"""
        g.E().hasLabel('{label}').limit(1)
         .properties()
         .group()
           .by(key)
           .by(
             project('类型', '示例值', '非空数')
               .by(value().label())
               .by(value())
               .by(
                 g.E().hasLabel('{label}').has(key).count()
               )
           )
        """
        return self.submit(gremlin)

    def get_doctor_edges_summary(self, doctor_id: str):
        """获取医生的边统计"""
        gremlin = f"""
        g.V().has('医生', '医生ID', '{doctor_id}').as('doc')
         .bothE().as('e')
         .select('doc', 'e')
         .by('姓名')
         .by(label)
         .group().by('doc').by(
           group().by('e').by(count())
         )
        """
        return self.submit(gremlin)

    def get_patient_treatment_timeline(self, patient_id: str):
        """获取患者治疗时间线"""
        gremlin = f"""
        g.V().has('患者', '患者ID', '{patient_id}')
         .bothE().hasLabel(within('治疗', '会诊', '手术', '医嘱'))
         .project('关系类型', '时间', '对方', '详情')
         .by(label)
         .by(coalesce(
           values('开始时间'),
           values('会诊时间'),
           values('手术时间'),
           values('开嘱时间'),
           constant('')
         ))
         .by(
           coalesce(
             outV().values('姓名'),
             inV().values('姓名')
           )
         )
         .by(valueMap())
         .order().by('时间', asc)
        """
        return self.submit(gremlin)

    def find_expensive_treatments(self, min_cost: float = 10000):
        """查找高费用治疗"""
        gremlin = f"""
        g.E().hasLabel('治疗')
         .has('治疗费用', gt({min_cost}))
         .project('医生', '患者', '费用', '类型', '状态')
         .by(outV().values('姓名'))
         .by(inV().values('姓名'))
         .by('治疗费用')
         .by('治疗类型')
         .by('治疗状态')
         .order().by('治疗费用', desc)
        """
        return self.submit(gremlin)

    def get_hospital_structure(self, hospital_id: str):
        """获取医院结构"""
        gremlin = f"""
        g.V().has('医院', '医院ID', '{hospital_id}')
         .project('医院', '科室数', '医生数', '患者数')
         .by('医院名称')
         .by(out('包含').count())
         .by(out('包含').in('属于').dedup().count())
         .by(out('包含').in('属于').out('治疗').dedup().count())
        """
        return self.submit(gremlin)


# =========================
# 数据初始化
# =========================

def initialize_complete_dataset():
    """初始化完整数据集"""
    system = MelanomaGraphSystem()
    try:
        print("=" * 70)
        print("开始初始化医疗知识图谱数据")
        print("=" * 70)

        # 1. 创建概念
        print("\n1. 创建概念层...")
        system.create_schema()

        # 2. 创建医院
        print("\n2. 创建医院...")
        hospitals = [
            {
                "hospital_id": "H001",
                "name": "北京协和医院",
                "level": "三甲",
                "city": "北京",
                "address": "北京市东城区帅府园1号",
                "phone": "010-69151188"
            },
            {
                "hospital_id": "H002",
                "name": "上海瑞金医院",
                "level": "三甲",
                "city": "上海",
                "address": "上海市黄浦区瑞金二路197号",
                "phone": "021-64370045"
            },
            {
                "hospital_id": "H003",
                "name": "广州中山医院",
                "level": "三甲",
                "city": "广州",
                "address": "广州市越秀区中山二路58号",
                "phone": "020-87332200"
            }
        ]

        for h in hospitals:
            system.create_hospital(**h)

        # 3. 创建科室
        print("\n3. 创建科室...")
        departments = [
            # 协和医院
            {"dept_id": "DEPT001", "name": "皮肤科", "dept_type": "临床", "director": "李主任", "phone": "010-69151188-1001",
             "bed_count": 50},
            {"dept_id": "DEPT002", "name": "肿瘤内科", "dept_type": "临床", "director": "王主任", "phone": "010-69151188-1002",
             "bed_count": 80},
            {"dept_id": "DEPT003", "name": "病理科", "dept_type": "医技", "director": "张主任", "phone": "010-69151188-2001",
             "bed_count": 0},
            {"dept_id": "DEPT004", "name": "放射科", "dept_type": "医技", "director": "刘主任", "phone": "010-69151188-2002",
             "bed_count": 0},

            # 瑞金医院
            {"dept_id": "DEPT005", "name": "皮肤科", "dept_type": "临床", "director": "陈主任", "phone": "021-64370045-1001",
             "bed_count": 60},
            {"dept_id": "DEPT006", "name": "肿瘤中心", "dept_type": "临床", "director": "赵主任", "phone": "021-64370045-1002",
             "bed_count": 100},

            # 中山医院
            {"dept_id": "DEPT007", "name": "皮肤科", "dept_type": "临床", "director": "孙主任", "phone": "020-87332200-1001",
             "bed_count": 55},
            {"dept_id": "DEPT008", "name": "肿瘤内科", "dept_type": "临床", "director": "周主任", "phone": "020-87332200-1002",
             "bed_count": 75}
        ]

        for d in departments:
            system.create_department(**d)

        # 4. 关联医院-科室
        print("\n4. 关联医院-科室...")
        hospital_dept_mappings = [
            ("H001", "DEPT001"), ("H001", "DEPT002"), ("H001", "DEPT003"), ("H001", "DEPT004"),
            ("H002", "DEPT005"), ("H002", "DEPT006"),
            ("H003", "DEPT007"), ("H003", "DEPT008")
        ]

        for h_id, d_id in hospital_dept_mappings:
            system.bind_hospital_department(h_id, d_id)

        # 5. 创建医生
        print("\n5. 创建医生...")
        doctors = [
            # 协和医院医生
            {
                "doctor_id": "DOC001", "name": "张明远", "gender": "男", "age": 45,
                "title": "主任医师", "specialty": "黑色素瘤、皮肤肿瘤", "years_exp": 20,
                "dept_id": "DEPT001", "phone": "13800138001", "email": "zhangmy@hospital.com"
            },
            {
                "doctor_id": "DOC002", "name": "王伟", "gender": "男", "age": 52,
                "title": "主任医师", "specialty": "肿瘤内科、靶向治疗", "years_exp": 25,
                "dept_id": "DEPT002", "phone": "13800138002", "email": "wangw@hospital.com"
            },
            {
                "doctor_id": "DOC003", "name": "李静", "gender": "女", "age": 38,
                "title": "副主任医师", "specialty": "皮肤病理、分子诊断", "years_exp": 12,
                "dept_id": "DEPT003", "phone": "13800138003", "email": "lij@hospital.com"
            },
            {
                "doctor_id": "DOC004", "name": "刘建国", "gender": "男", "age": 48,
                "title": "主任医师", "specialty": "放射诊断", "years_exp": 22,
                "dept_id": "DEPT004", "phone": "13800138004", "email": "liujg@hospital.com"
            },

            # 瑞金医院医生
            {
                "doctor_id": "DOC005", "name": "陈建国", "gender": "男", "age": 48,
                "title": "主任医师", "specialty": "皮肤外科、Mohs手术", "years_exp": 22,
                "dept_id": "DEPT005", "phone": "13800138005", "email": "chenjg@hospital.com"
            },
            {
                "doctor_id": "DOC006", "name": "刘芳", "gender": "女", "age": 42,
                "title": "副主任医师", "specialty": "免疫治疗、临床研究", "years_exp": 15,
                "dept_id": "DEPT006", "phone": "13800138006", "email": "liuf@hospital.com"
            },

            # 中山医院医生
            {
                "doctor_id": "DOC007", "name": "孙明", "gender": "男", "age": 50,
                "title": "主任医师", "specialty": "皮肤肿瘤", "years_exp": 24,
                "dept_id": "DEPT007", "phone": "13800138007", "email": "sunm@hospital.com"
            },
            {
                "doctor_id": "DOC008", "name": "周丽", "gender": "女", "age": 39,
                "title": "副主任医师", "specialty": "肿瘤化疗", "years_exp": 13,
                "dept_id": "DEPT008", "phone": "13800138008", "email": "zhoul@hospital.com"
            }
        ]

        for doctor in doctors:
            system.create_doctor(**doctor)

        # 6. 创建患者
        print("\n6. 创建患者...")
        patients = [
            {
                "patient_id": "P001", "name": "张三", "gender": "男", "age": 55,
                "id_card": "110101196801015678", "phone": "13900139001",
                "address": "北京市朝阳区", "blood_type": "A", "admission_date": "2023-03-15"
            },
            {
                "patient_id": "P002", "name": "李四", "gender": "女", "age": 68,
                "id_card": "110101195501015679", "phone": "13900139002",
                "address": "北京市海淀区", "blood_type": "O", "admission_date": "2023-04-20"
            },
            {
                "patient_id": "P003", "name": "王五", "gender": "男", "age": 50,
                "id_card": "310101197301015680", "phone": "13900139003",
                "address": "上海市徐汇区", "blood_type": "B", "admission_date": "2023-05-10"
            },
            {
                "patient_id": "P004", "name": "赵六", "gender": "女", "age": 62,
                "id_card": "310101196101015681", "phone": "13900139004",
                "address": "上海市静安区", "blood_type": "AB", "admission_date": "2023-06-05"
            },
            {
                "patient_id": "P005", "name": "钱七", "gender": "男", "age": 45,
                "id_card": "440101197801015682", "phone": "13900139005",
                "address": "广州市天河区", "blood_type": "A", "admission_date": "2023-07-12"
            },
            {
                "patient_id": "P006", "name": "孙八", "gender": "女", "age": 58,
                "id_card": "440101196501015683", "phone": "13900139006",
                "address": "广州市越秀区", "blood_type": "O", "admission_date": "2023-08-18"
            }
        ]

        for p in patients:
            system.create_patient(**p)

        # 7. 创建治疗关系（带丰富属性）
        print("\n7. 创建治疗关系...")
        treatments = [
            # 张明远医生的治疗
            ("DOC001", "P001", "门诊", "2023-03-15", "2023-06-15", "完成", 15000.0, 0.8, "定期复查"),
            ("DOC001", "P002", "住院", "2023-04-20", None, "进行中", 35000.0, 0.7, "术后恢复期"),

            # 王伟医生的治疗
            ("DOC002", "P003", "门诊", "2023-05-10", "2023-08-10", "完成", 12000.0, 0.75, "化疗后随访"),

            # 陈建国医生的治疗
            ("DOC005", "P004", "住院", "2023-06-05", "2023-09-05", "完成", 42000.0, 0.65, "手术成功"),

            # 孙明医生的治疗
            ("DOC007", "P005", "门诊", "2023-07-12", None, "进行中", 8000.0, 0.8, "新患者初诊"),

            # 周丽医生的治疗
            ("DOC008", "P006", "住院", "2023-08-18", None, "进行中", 28000.0, 0.7, "化疗第一周期")
        ]

        for treatment in treatments:
            system.create_treatment_relationship(*treatment)

        # 8. 创建药品
        print("\n8. 创建药品...")
        drugs = [
            ("维莫非尼", "靶向药物", "罗氏制药", "BRAF V600突变阳性的黑色素瘤", 800.0),
            ("帕博利珠单抗", "免疫治疗药物", "默沙东", "晚期黑色素瘤", 18000.0),
            ("达拉非尼", "靶向药物", "诺华", "BRAF突变阳性黑色素瘤", 750.0),
            ("曲美替尼", "靶向药物", "诺华", "BRAF突变阳性黑色素瘤", 820.0),
            ("顺铂", "化疗药物", "多家药企", "多种实体瘤", 50.0),
            ("紫杉醇", "化疗药物", "多家药企", "多种实体瘤", 120.0)
        ]

        for drug in drugs:
            system.create_drug(*drug)

        # 9. 创建处方
        print("\n9. 创建处方...")
        prescriptions = [
            ("RX001", "DOC001", "P001", "2023-03-16", "已完成"),
            ("RX002", "DOC001", "P002", "2023-04-21", "待取药"),
            ("RX003", "DOC002", "P003", "2023-05-11", "已完成"),
            ("RX004", "DOC005", "P004", "2023-06-06", "已完成"),
            ("RX005", "DOC007", "P005", "2023-07-13", "待取药")
        ]

        for rx in prescriptions:
            system.create_prescription(*rx)

        # 10. 处方中添加药品
        print("\n10. 处方中添加药品...")
        prescription_drugs = [
            ("RX001", "维莫非尼", "960mg", "每日两次", "3个月", 90),
            ("RX001", "顺铂", "50mg/m²", "每三周一次", "3周期", 3),
            ("RX002", "帕博利珠单抗", "200mg", "每三周一次", "6个月", 8),
            ("RX003", "达拉非尼", "150mg", "每日两次", "3个月", 90),
            ("RX003", "曲美替尼", "2mg", "每日一次", "3个月", 90),
            ("RX004", "维莫非尼", "960mg", "每日两次", "6个月", 180),
            ("RX005", "紫杉醇", "175mg/m²", "每三周一次", "3周期", 3)
        ]

        for pd in prescription_drugs:
            system.add_drug_to_prescription(*pd)

        # 11. 创建会诊关系
        print("\n11. 创建会诊关系...")
        consultations = [
            ("DOC002", "P001", "多学科会诊", "2023-03-28", 120,
             ["张明远", "王伟", "李静"], "建议靶向治疗联合手术", 5000.0),
            ("DOC001", "P002", "院内会诊", "2023-05-10", 90,
             ["张明远", "刘建国"], "建议放疗后手术", 3000.0),
            ("DOC006", "P004", "远程会诊", "2023-07-15", 60,
             ["陈建国", "刘芳"], "确认免疫治疗方案", 2000.0)
        ]

        for consultation in consultations:
            system.create_consultation(*consultation)

        # 12. 创建手术关系
        print("\n12. 创建手术关系...")
        surgeries = [
            ("DOC001", "P001", "黑色素瘤扩大切除术", "2023-06-20", 180,
             "全身麻醉", True, 25000.0),
            ("DOC005", "P004", "Mohs显微描记手术", "2023-07-10", 240,
             "局部麻醉", True, 32000.0),
            ("DOC007", "P005", "前哨淋巴结活检", "2023-08-05", 90,
             "局部麻醉", True, 15000.0)
        ]

        for surgery in surgeries:
            system.create_surgery_relationship(*surgery)

        # 13. 创建医嘱关系
        print("\n13. 创建医嘱关系...")
        medical_orders = [
            ("DOC001", "P001", "出院医嘱", "1. 按时服药；2. 定期复查；3. 注意饮食",
             "2023-06-30", "护士A", "已执行"),
            ("DOC001", "P002", "住院医嘱", "1. 每日测量体温；2. 伤口护理；3. 限制活动",
             "2023-05-05", "护士B", "已执行"),
            ("DOC005", "P004", "术后医嘱", "1. 伤口保持干燥；2. 避免剧烈运动；3. 按时复查",
             "2023-07-11", "护士C", "已执行")
        ]

        for order in medical_orders:
            system.create_medical_order(*order)

        # 14. 创建医患沟通关系
        print("\n14. 创建医患沟通关系...")
        communications = [
            ("DOC001", "P001", "门诊", "2023-07-10", 30,
             "复查结果说明", "理解治疗方案", 5),
            ("DOC001", "P002", "电话", "2023-08-15", 20,
             "病情咨询", "满意解答", 4),
            ("DOC005", "P004", "门诊", "2023-08-20", 25,
             "术后恢复情况", "满意服务", 5)
        ]

        for comm in communications:
            system.create_communication(*comm)

        # 15. 创建转诊关系
        print("\n15. 创建转诊关系...")
        referrals = [
            ("DOC001", "DOC003", "P001", "需要病理科专家会诊", "2023-03-25", "科间转诊"),
            ("DOC005", "DOC006", "P004", "需要免疫治疗专家意见", "2023-07-05", "科间转诊"),
            ("DOC007", "DOC008", "P005", "需要化疗专家评估", "2023-08-01", "科间转诊")
        ]

        for referral in referrals:
            system.create_referral(*referral)

        # 16. 创建同事关系
        print("\n16. 创建同事关系...")
        colleagues = [
            ("DOC001", "DOC002", 12, "2023-07-01", "长期合作"),
            ("DOC001", "DOC003", 8, "2023-06-15", "科室合作"),
            ("DOC005", "DOC006", 15, "2023-08-10", "跨科合作"),
            ("DOC007", "DOC008", 6, "2023-07-20", "新合作")
        ]

        for colleague in colleagues:
            system.create_colleague_relationship(*colleague)

        # 17. 创建样本和基因突变
        print("\n17. 创建样本和基因突变...")

        # 样本
        samples = [
            ("SAMP001", "P001", "组织活检", "2023-03-16"),
            ("SAMP002", "P002", "血液样本", "2023-04-21"),
            ("SAMP003", "P004", "组织活检", "2023-06-06"),
            ("SAMP004", "P005", "组织活检", "2023-07-13")
        ]

        for sample in samples:
            system.create_sample(*sample)

        # 基因突变
        mutations = [
            ("BRAF", "SAMP001", "V600E", 0.45),
            ("NRAS", "SAMP001", "Q61K", 0.12),
            ("BRAF", "SAMP003", "V600K", 0.38),
            ("NRAS", "SAMP004", "Q61R", 0.25)
        ]

        for mutation in mutations:
            system.create_gene_mutation(*mutation)

        # 18. 创建随访记录
        print("\n18. 创建随访记录...")
        followups = [
            ("FU001", "P001", "DOC001", "2023-06-30", "存活", "否",
             "患者一般情况良好，无不适主诉"),
            ("FU002", "P002", "DOC001", "2023-07-15", "存活", "是",
             "发现局部淋巴结转移，建议二次手术"),
            ("FU003", "P003", "DOC002", "2023-08-10", "存活", "否",
             "化疗后恢复良好，无不良反应"),
            ("FU004", "P004", "DOC005", "2023-09-05", "存活", "否",
             "术后恢复良好，伤口愈合正常")
        ]

        for fu in followups:
            system.create_followup(*fu)

        print("\n" + "=" * 70)
        print("数据初始化完成!")
        print("=" * 70)
        print(f"创建统计:")
        print(f"  - 医院: {len(hospitals)} 家")
        print(f"  - 科室: {len(departments)} 个")
        print(f"  - 医生: {len(doctors)} 名")
        print(f"  - 患者: {len(patients)} 名")
        print(f"  - 药品: {len(drugs)} 种")
        print(f"  - 处方: {len(prescriptions)} 张")
        print(f"  - 治疗关系: {len(treatments)} 条")
        print(f"  - 会诊关系: {len(consultations)} 条")
        print(f"  - 手术关系: {len(surgeries)} 条")
        print(f"  - 医嘱关系: {len(medical_orders)} 条")
        print(f"  - 随访记录: {len(followups)} 条")
        print("=" * 70)

    finally:
        system.close()

    return True


# =========================
# 边查询演示系统
# =========================

class EdgeQueryDemo:
    """边查询演示类"""

    def __init__(self, system):
        self.system = system

    def demo_basic_edge_queries(self):
        """基础边查询演示"""
        print("\n" + "=" * 70)
        print("1. 基础边查询演示")
        print("=" * 70)

        # 1.1 查询所有边
        print("\n1.1 查询所有边（前5条）:")
        gremlin = "g.E().elementMap()"
        edges = self.system.submit(gremlin)
        for i, edge in enumerate(edges):
            print(f"  边{i + 1}: 标签={edge.get(T.label, '未知')}, "
                  f"ID={edge.get(T.id, '未知')}")

        # 1.2 查询特定标签的边
        print("\n1.2 查询'治疗'边:")
        gremlin = "g.E().hasLabel('治疗').limit(5).elementMap()"
        edges = self.system.submit(gremlin)
        print(edges)
        for i, edge in enumerate(edges):
            print(f"  治疗边{i + 1}:")
            print(f"    起点: {edge.get(Direction.OUT, '未知')}")
            print(f"    终点: {edge.get(Direction.IN, '未知')}")
            print(f"    费用: {edge.get('治疗费用', '未知')}")
            print(f"    状态: {edge.get('治疗状态', '未知')}")
        #
        # 1.3 统计边数量
        print("\n1.3 边类型统计:")
        gremlin = "g.E().groupCount().by(label)"
        result = self.system.submit(gremlin)
        print(result)
        if result:
            edge_counts = result[0]
            for label, count in sorted(edge_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {label}: {count}条")

    def demo_edge_property_queries(self):
        """边属性查询演示"""
        print("\n" + "=" * 70)
        print("2. 边属性查询演示")
        print("=" * 70)

        # 2.1 查询有特定属性的边
        print("\n2.1 查询有'治疗费用'属性的边:")
        gremlin = """
            g.E().hasLabel('治疗')
             .has('治疗费用')
             .limit(3)
             .project('医生', '患者', '费用')
             .by(outV().values('姓名'))
             .by(inV().values('姓名'))
             .by('治疗费用')
        """
        edges = self.system.submit(gremlin)
        print(edges)
        for edge in edges:
            print(f"  {edge['医生']} -> {edge['患者']}: ¥{edge['费用']}")

        # 2.2 按属性值范围查询
        # print("\n2.2 查询费用大于20000的治疗:")
        # gremlin = """
        # g.E().hasLabel('治疗')
        #  .has('治疗费用', gt(20000))
        #  .project('医生', '患者', '费用', '类型')
        #  .by(outV().values('姓名'))
        #  .by(inV().values('姓名'))
        #  .by('治疗费用')
        #  .by('治疗类型')
        #  .order().by('治疗费用', desc)
        # """
        # edges = self.system.submit(gremlin)
        # for edge in edges:
        #     print(f"  {edge['医生']} -> {edge['患者']}: "
        #           f"¥{edge['费用']} ({edge['类型']})")
        #
        # # 2.3 按时间范围查询
        # print("\n2.3 查询2023年第二季度的治疗:")
        # gremlin = """
        # g.E().hasLabel('治疗')
        #  .has('开始时间', between('2023-04-01', '2023-06-30'))
        #  .project('医生', '患者', '开始时间', '状态')
        #  .by(outV().values('姓名'))
        #  .by(inV().values('姓名'))
        #  .by('开始时间')
        #  .by('治疗状态')
        #  .order().by('开始时间', asc)
        # """
        # edges = self.system.submit(gremlin)
        # for edge in edges:
        #     print(f"  {edge['开始时间']}: {edge['医生']} -> {edge['患者']} "
        #           f"({edge['状态']})")

    def demo_edge_vertex_relationship(self):
        """边与顶点关系演示"""
        print("\n" + "=" * 70)
        print("3. 边与顶点关系演示")
        print("=" * 70)

        # 3.1 查询边的起点和终点
        print("\n3.1 查询边的完整连接关系:")
        gremlin = """
        g.E().hasLabel('治疗').limit(3)
         .project('边ID', '起点', '起点类型', '终点', '终点类型', '边属性')
         .by(id())
         .by(outV().valueMap('姓名', '医生ID'))
         .by(outV().label())
         .by(inV().valueMap('姓名', '患者ID'))
         .by(inV().label())
         .by(valueMap('治疗类型', '治疗状态', '治疗费用'))
        """
        edges = self.system.submit(gremlin)
        for edge in edges:
            print(f"\n  边ID: {edge['边ID'][:20]}...")
            print(f"  起点: {edge['起点'][0].get('姓名', '未知')} "
                  f"({edge['起点类型']})")
            print(f"  终点: {edge['终点'][0].get('姓名', '未知')} "
                  f"({edge['终点类型']})")
            print(f"  属性: {edge['边属性']}")

        # 3.2 查询特定顶点的所有边
        print("\n3.2 查询医生'张明远'的所有关系:")
        gremlin = """
        g.V().has('医生', '姓名', '张明远').as('doctor')
         .bothE().as('edge')
         .otherV().as('other')
         .select('doctor', 'edge', 'other')
         .by('姓名')
         .by(label)
         .by(
           coalesce(
             values('姓名'),
             values('患者ID'),
             values('药物名称'),
             label()
           )
         )
         .limit(10)
        """
        edges = self.system.submit(gremlin)
        for edge in edges:
            print(f"  张明远 --[{edge['edge']}]--> {edge['other']}")

    def demo_edge_statistics(self):
        """边统计演示"""
        print("\n" + "=" * 70)
        print("4. 边统计演示")
        print("=" * 70)

        # 4.1 边类型分布
        print("\n4.1 边类型分布统计:")
        stats = self.system.get_edge_statistics()
        if stats:
            for item in stats[:10]:  # 显示前10种
                key = item[0]
                count = item[1]
                print(f"  {key['起点标签']}--[{key['边标签']}]-->"
                      f"{key['终点标签']}: {count}条")

        # 4.2 医生边统计
        print("\n4.2 医生关系统计:")
        gremlin = """
        g.V().hasLabel('医生').as('doc')
         .project('医生', '科室', '治疗数', '会诊数', '医嘱数')
         .by('姓名')
         .by(out('属于').values('科室名称'))
         .by(outE('治疗').count())
         .by(outE('会诊').count())
         .by(outE('医嘱').count())
         .order().by('治疗数', desc)
        """
        stats = self.system.submit(gremlin)
        for stat in stats:
            print(f"  {stat['医生']} ({stat['科室'][0]}): "
                  f"治疗{stat['治疗数']}次, "
                  f"会诊{stat['会诊数']}次, "
                  f"医嘱{stat['医嘱数']}次")

        # 4.3 治疗费用统计
        print("\n4.3 治疗费用统计:")
        gremlin = """
        g.E().hasLabel('治疗')
         .has('治疗费用')
         .project('总计', '平均', '最高', '最低')
         .by(values('治疗费用').sum())
         .by(values('治疗费用').mean())
         .by(values('治疗费用').max())
         .by(values('治疗费用').min())
        """
        stats = self.system.submit(gremlin)
        if stats:
            stat = stats[0]
            print(f"  总计费用: ¥{stat['总计']:.2f}")
            print(f"  平均费用: ¥{stat['平均']:.2f}")
            print(f"  最高费用: ¥{stat['最高']:.2f}")
            print(f"  最低费用: ¥{stat['最低']:.2f}")

    def demo_edge_path_queries(self):
        """边路径查询演示"""
        print("\n" + "=" * 70)
        print("5. 边路径查询演示")
        print("=" * 70)

        # 5.1 医院-科室-医生路径
        print("\n5.1 医院组织结构路径:")
        gremlin = """
        g.V().has('医院', '医院名称', '北京协和医院')
         .outE('包含').as('e1')
         .inV().hasLabel('科室').as('dept')
         .inE('属于').as('e2')
         .outV().hasLabel('医生').as('doctor')
         .select('e1', 'dept', 'e2', 'doctor')
         .by(valueMap())
         .by('科室名称')
         .by(valueMap())
         .by('姓名')
         .limit(3)
        """
        paths = self.system.submit(gremlin)
        for path in paths:
            print(f"\n  医院->科室边: {path['e1']}")
            print(f"  科室: {path['dept']}")
            print(f"  科室->医生边: {path['e2']}")
            print(f"  医生: {path['doctor']}")

        # 5.2 医生-患者-样本-基因突变路径
        print("\n5.2 医疗数据完整路径:")
        gremlin = """
        g.V().has('医生', '姓名', '张明远')
         .outE('治疗').as('treat')
         .inV().hasLabel('患者').as('patient')
         .outE('拥有').as('own')
         .inV().hasLabel('样本').as('sample')
         .outE('包含').as('contain')
         .inV().hasLabel('基因突变').as('mutation')
         .path()
         .by('姓名')
         .by(valueMap('治疗类型', '治疗状态'))
         .by('姓名')
         .by(valueMap('采集日期'))
         .by('样本ID')
         .by(valueMap())
         .by('基因名')
         .limit(2)
        """
        paths = self.system.submit(gremlin)
        for path in paths:
            print(f"\n  完整路径:")
            for i, step in enumerate(path):
                print(f"    步骤{i}: {step}")

    def demo_edge_operations(self):
        """边操作演示"""
        print("\n" + "=" * 70)
        print("6. 边操作演示")
        print("=" * 70)

        # 6.1 查询边属性摘要
        print("\n6.1 治疗边的属性摘要:")
        summary = self.system.get_edge_properties_summary("治疗")
        if summary:
            for prop_name, info in summary[0].items():
                print(f"  {prop_name}: 类型={info['类型']}, "
                      f"示例={info['示例值']}, 非空数={info['非空数']}")

        # 6.2 查找重复边（演示查询）
        print("\n6.2 查找重复边模式:")
        gremlin = """
        g.E().group()
         .by(
           project('from', 'to', 'label')
             .by(outV().id())
             .by(inV().id())
             .by(label)
         )
         .by(count())
         .unfold()
         .where(select(values).is(gt(1)))
         .select(keys)
         .limit(5)
        """
        duplicates = self.system.submit(gremlin)
        if duplicates:
            print(f"  找到重复边: {len(duplicates)} 组")
        else:
            print("  没有找到重复边")

        # 6.3 边的更新操作（演示）
        print("\n6.3 边更新操作（演示查询）:")
        # 这里只演示查询，不实际更新
        gremlin = """
        g.E().hasLabel('治疗')
         .has('治疗状态', '进行中')
         .limit(1)
         .project('更新前状态', '更新操作')
         .by('治疗状态')
         .by(constant('可更新为"完成"'))
        """
        updates = self.system.submit(gremlin)
        for update in updates:
            print(f"  边状态: {update['更新前状态']}, {update['更新操作']}")

    def run_all_demos(self):
        """运行所有演示"""

        # self.demo_basic_edge_queries()
        self.demo_edge_property_queries()
        # self.demo_edge_vertex_relationship()
        # self.demo_edge_statistics()
        # self.demo_edge_path_queries()
        # self.demo_edge_operations()

        print("\n" + "=" * 70)
        print("边查询演示完成")
        print("=" * 70)


# =========================
# 实用场景查询
# =========================

def practical_scenarios_demo():
    """实用场景演示"""
    system = MelanomaGraphSystem()
    try:
        print("\n" + "=" * 70)
        print("实用场景演示")
        print("=" * 70)

        # 场景1：医疗费用分析
        print("\n场景1: 医疗费用分析")
        print("-" * 40)

        expensive_treatments = system.find_expensive_treatments(20000)
        if expensive_treatments:
            print("  高费用治疗记录:")
            for treatment in expensive_treatments:
                print(f"    {treatment['医生']} -> {treatment['患者']}: "
                      f"¥{treatment['费用']} ({treatment['类型']}, {treatment['状态']})")

        # 场景2：患者治疗时间线
        print("\n场景2: 患者治疗时间线")
        print("-" * 40)

        timeline = system.get_patient_treatment_timeline("P001")
        if timeline:
            print("  患者P001治疗时间线:")
            for event in timeline:
                print(f"    {event['时间']}: {event['关系类型']} "
                      f"({event['对方']})")
                # 显示部分详情
                details = event['详情']
                if '治疗类型' in details:
                    print(f"      类型: {details['治疗类型'][0]}, "
                          f"状态: {details.get('治疗状态', [''])[0]}")
                elif '会诊类型' in details:
                    print(f"      类型: {details['会诊类型'][0]}")

        # 场景3：医生工作量分析
        print("\n场景3: 医生工作量分析")
        print("-" * 40)

        doctors_workload = system.submit("""
        g.V().hasLabel('医生')
         .project('医生', '科室', '治疗数', '会诊数', '手术数', '医嘱数')
         .by('姓名')
         .by(out('属于').values('科室名称'))
         .by(outE('治疗').count())
         .by(outE('会诊').count())
         .by(outE('手术').count())
         .by(outE('医嘱').count())
         .order().by('治疗数', desc)
        """)

        for doctor in doctors_workload:
            print(f"  {doctor['医生']} ({doctor['科室'][0]}): "
                  f"治疗{doctor['治疗数']}次, "
                  f"会诊{doctor['会诊数']}次, "
                  f"手术{doctor['手术数']}次, "
                  f"医嘱{doctor['医嘱数']}次")

        # 场景4：医院结构分析
        print("\n场景4: 医院结构分析")
        print("-" * 40)

        hospital_structure = system.get_hospital_structure("H001")
        if hospital_structure:
            stats = hospital_structure[0]
            print(f"  医院: {stats['医院']}")
            print(f"  科室数: {stats['科室数']}")
            print(f"  医生数: {stats['医生数']}")
            print(f"  患者数: {stats['患者数']}")

        # 场景5：药品使用分析
        print("\n场景5: 药品使用分析")
        print("-" * 40)

        drug_usage = system.submit("""
        g.V().hasLabel('药物')
         .project('药品名称', '使用处方数', '单价', '生产厂家')
         .by('药物名称')
         .by(inE('包含药品').count())
         .by('单价')
         .by('生产厂家')
         .order().by('使用处方数', desc)
        """)

        for drug in drug_usage:
            print(f"  {drug['药品名称']}: "
                  f"使用处方{drug['使用处方数']}张, "
                  f"单价¥{drug['单价']}, "
                  f"生产商:{drug['生产厂家']}")

        print("\n" + "=" * 70)
        print("实用场景演示完成")
        print("=" * 70)

    finally:
        system.close()


# =========================
# 主程序
# =========================

def main():
    """主程序"""
    # initialize_complete_dataset()
    system = MelanomaGraphSystem()
    try:
        demo = EdgeQueryDemo(system)
        demo.run_all_demos()
    finally:
        system.close()

    # practical_scenarios_demo()
    #
    # system = MelanomaGraphSystem()
    # try:
    #     print("\n快速查询测试...")
    #
    #     # 测试1：基本统计
    #     print("\n1. 基本统计:")
    #     total_vertices = system.submit("g.V().count()")[0]
    #     total_edges = system.submit("g.E().count()")[0]
    #     print(f"   总顶点数: {total_vertices}")
    #     print(f"   总边数: {total_edges}")
    #
    #     # 测试2：边类型
    #     print("\n2. 边类型分布:")
    #     edge_stats = system.get_edge_statistics()
    #     if edge_stats:
    #         for item in edge_stats[:5]:
    #             key = item[0]
    #             count = item[1]
    #             print(f"   {key['起点标签']}--[{key['边标签']}]-->"
    #                   f"{key['终点标签']}: {count}条")
    #
    #     # 测试3：查询特定边
    #     print("\n3. 治疗边示例:")
    #     treatments = system.get_edges_by_label("治疗", limit=3)
    #     for i, edge in enumerate(treatments):
    #         out_v = edge.get('outV', '未知')
    #         in_v = edge.get('inV', '未知')
    #         print(f"   边{i + 1}: {out_v} -> {in_v}")

    # finally:
    #     system.close()



# =========================
# 程序入口
# =========================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    finally:
        print("程序结束")