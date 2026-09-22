"""
3D Bin Packing Prototype
=========================
Heuristic: Extreme Point-based placement (คล้าย Deepest-Bottom-Left-Fill)

แนวคิด:
- เรียงสินค้าตามปริมาตร (ใหญ่ไปเล็ก) ก่อนเสมอ เพื่อวางของใหญ่ก่อนแล้วอุดช่องว่างด้วยของเล็ก
- เก็บรายการ "Extreme Points" (จุดที่เป็นไปได้สำหรับวางกล่องถัดไป) ไว้เป็น candidate positions
- สำหรับสินค้าแต่ละชิ้น ลองทุก orientation (ถ้าอนุญาตให้หมุน) ที่ทุก extreme point
  แล้วเลือกตำแหน่งที่ทำให้ของอยู่ "ต่ำสุด-ซ้ายสุด-ลึกสุด" ก่อน (ลด void space)
- ตรวจสอบ: ไม่ชนกล่องอื่น, ไม่เกินขอบตู้, ไม่เกินน้ำหนักรวม, มีฐานรองรับน้ำหนักเพียงพอ

ใช้งานได้จริงระดับ prototype/MVP — เหมาะสำหรับทดสอบแนวคิดก่อนต่อยอดเป็น
Genetic Algorithm / Reinforcement Learning / MIP solver ในสเต็ปถัดไป
"""

from dataclasses import dataclass, field
from itertools import permutations
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Item:
    id: str
    length: float
    width: float
    height: float
    weight: float
    allow_rotation: bool = True   # หมุนกล่องได้ไหม (สินค้าบางชนิดต้องตั้งด้านเดิมเสมอ)
    fragile: bool = False         # ห้ามมีของวางทับด้านบน
    quantity: int = 1             # จำนวนของ item นี้

    # ค่าที่ถูกกำหนดหลัง pack สำเร็จ
    position: Optional[Tuple[float, float, float]] = None
    dims_packed: Optional[Tuple[float, float, float]] = None

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    def orientations(self):
        """คืนค่า dimension ที่เป็นไปได้ทั้งหมด (ถ้าหมุนได้ คืน 6 แบบ ถ้าไม่ คืนแบบเดียว)"""
        dims = (self.length, self.width, self.height)
        if not self.allow_rotation:
            return [dims]
        return list(set(permutations(dims)))


@dataclass
class Container:
    id: str
    length: float
    width: float
    height: float
    max_weight: float

    placed_items: List[Item] = field(default_factory=list)
    current_weight: float = 0.0
    # Extreme points เริ่มต้น: มุมล่างซ้ายหน้าของตู้
    extreme_points: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [(0.0, 0.0, 0.0)]
    )

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    def used_volume(self) -> float:
        return sum(i.volume for i in self.placed_items)

    def utilization(self) -> float:
        return self.used_volume() / self.volume * 100 if self.volume else 0.0


# ---------------------------------------------------------------------------
# Core packing logic
# ---------------------------------------------------------------------------

def _overlaps(pos_a, dim_a, pos_b, dim_b) -> bool:
    """เช็คว่ากล่อง 2 ใบทับซ้อนกันในพื้นที่ 3 มิติหรือไม่"""
    ax, ay, az = pos_a
    al, aw, ah = dim_a
    bx, by, bz = pos_b
    bl, bw, bh = dim_b

    return not (
        ax + al <= bx or bx + bl <= ax or
        ay + aw <= by or by + bw <= ay or
        az + ah <= bz or bz + bh <= az
    )


def _has_support(pos, dim, placed_items, container) -> bool:
    """เช็คว่าฐานของกล่องมีอะไรรองรับเพียงพอ (พื้นตู้ หรือกล่องอื่นที่สูงพอดี)"""
    x, y, z = pos
    l, w, h = dim

    if z == 0:
        return True  # วางบนพื้นตู้โดยตรง

    supported_area = 0.0
    footprint = l * w

    for other in placed_items:
        ox, oy, oz = other.position
        ol, ow, oh = other.dims_packed
        if other.fragile:
            continue  # ของเปราะห้ามมีอะไรวางทับ
        # เช็คว่าหน้าตัดบนของ other อยู่ที่ระดับ z นี้พอดีไหม
        if abs((oz + oh) - z) > 1e-6:
            continue
        # คำนวณพื้นที่ที่ทับซ้อนกันในแนวราบ (x, y)
        overlap_x = max(0, min(x + l, ox + ol) - max(x, ox))
        overlap_y = max(0, min(y + w, oy + ow) - max(y, oy))
        supported_area += overlap_x * overlap_y

    # กำหนดเกณฑ์: ต้องมีฐานรองรับอย่างน้อย 70% ของพื้นที่ฐานกล่อง
    return supported_area >= 0.7 * footprint


def _fits_in_container(pos, dim, container) -> bool:
    x, y, z = pos
    l, w, h = dim
    return (
        x + l <= container.length + 1e-6 and
        y + w <= container.width + 1e-6 and
        z + h <= container.height + 1e-6
    )


def _generate_new_extreme_points(pos, dim, container):
    """หลังวางกล่องแล้ว สร้าง extreme point ใหม่ 3 จุด (ขวา, หลัง, บน ของกล่องที่เพิ่งวาง)"""
    x, y, z = pos
    l, w, h = dim
    return [
        (x + l, y, z),  # ถัดไปทางขวา
        (x, y + w, z),  # ถัดไปทางหลัง
        (x, y, z + h),  # ถัดไปด้านบน
    ]


def pack(container: Container, items: List[Item]) -> Tuple[List[Item], List[Item]]:
    """
    พยายามจัดวาง items ทั้งหมดลงใน container
    คืนค่า (placed, unplaced)
    """
    # ขยาย quantity ให้เป็น item เดี่ยวๆ ทีละชิ้น
    expanded: List[Item] = []
    for it in items:
        for n in range(it.quantity):
            expanded.append(Item(
                id=f"{it.id}#{n+1}" if it.quantity > 1 else it.id,
                length=it.length, width=it.width, height=it.height,
                weight=it.weight, allow_rotation=it.allow_rotation,
                fragile=it.fragile,
            ))

    # เรียงจากปริมาตรมากไปน้อย → วางของใหญ่ก่อนเสมอ (ลด void space)
    expanded.sort(key=lambda i: i.volume, reverse=True)

    unplaced: List[Item] = []

    for item in expanded:
        if container.current_weight + item.weight > container.max_weight:
            unplaced.append(item)
            continue

        best_point = None
        best_dim = None
        best_score = None  # ยิ่งน้อยยิ่งดี (z, y, x) → ต่ำสุด-ลึกสุด-ซ้ายสุดก่อน

        for point in container.extreme_points:
            for dim in item.orientations():
                if not _fits_in_container(point, dim, container):
                    continue
                if any(_overlaps(point, dim, o.position, o.dims_packed)
                       for o in container.placed_items):
                    continue
                if not _has_support(point, dim, container.placed_items, container):
                    continue

                score = (point[2], point[1], point[0])  # z, y, x
                if best_score is None or score < best_score:
                    best_score = score
                    best_point = point
                    best_dim = dim

        if best_point is None:
            unplaced.append(item)
            continue

        # ยืนยันการวาง
        item.position = best_point
        item.dims_packed = best_dim
        container.placed_items.append(item)
        container.current_weight += item.weight

        # อัปเดต extreme points: ลบจุดที่ใช้ไปแล้ว เพิ่มจุดใหม่
        container.extreme_points.remove(best_point)
        container.extreme_points.extend(
            _generate_new_extreme_points(best_point, best_dim, container)
        )

    return container.placed_items, unplaced


# ---------------------------------------------------------------------------
# Visualization (matplotlib 3D)
# ---------------------------------------------------------------------------

def visualize(container: Container, save_path: str = "packing_result.png"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import random

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    random.seed(42)

    def cuboid_faces(pos, dim):
        x, y, z = pos
        l, w, h = dim
        v = [
            [x, y, z], [x+l, y, z], [x+l, y+w, z], [x, y+w, z],
            [x, y, z+h], [x+l, y, z+h], [x+l, y+w, z+h], [x, y+w, z+h],
        ]
        return [
            [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
            [v[1], v[2], v[6], v[5]], [v[4], v[7], v[3], v[0]],
        ]

    for item in container.placed_items:
        color = (random.random(), random.random(), random.random())
        faces = cuboid_faces(item.position, item.dims_packed)
        poly = Poly3DCollection(faces, alpha=0.8, facecolor=color, edgecolor="black", linewidths=0.5)
        ax.add_collection3d(poly)

    ax.set_xlim(0, container.length)
    ax.set_ylim(0, container.width)
    ax.set_zlim(0, container.height)
    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    ax.set_zlabel("Height")
    ax.set_title(f"{container.id} — Utilization: {container.utilization():.1f}%")
    ax.set_box_aspect((container.length, container.width, container.height))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ตัวอย่างตู้คอนเทนเนอร์ขนาด 40ft (หน่วย: เมตร) — ปรับตามจริงได้
    container = Container(
        id="Container-40HC",
        length=12.03, width=2.35, height=2.69,
        max_weight=26000,  # kg
    )

    # ตัวอย่างรายการสินค้า (ปรับให้ตรงกับ SKU จริงของคุณ)
    items = [
        Item(id="BoxA", length=1.2, width=1.0, height=1.0, weight=300, quantity=10),
        Item(id="BoxB", length=0.8, width=0.6, height=0.6, weight=80, quantity=25),
        Item(id="BoxC", length=0.5, width=0.5, height=0.4, weight=20, quantity=40, fragile=True),
        Item(id="BoxD", length=2.0, width=1.0, height=0.5, weight=150, allow_rotation=False, quantity=8),
    ]

    placed, unplaced = pack(container, items)

    print(f"\n=== ผลการจัดเรียง: {container.id} ===")
    print(f"วางสำเร็จ : {len(placed)} ชิ้น")
    print(f"วางไม่ได้  : {len(unplaced)} ชิ้น")
    print(f"อัตราการใช้พื้นที่ (Volume Utilization): {container.utilization():.1f}%")
    print(f"น้ำหนักรวม: {container.current_weight:.1f} / {container.max_weight} kg "
          f"({container.current_weight/container.max_weight*100:.1f}%)")

    if unplaced:
        print("\nรายการที่วางไม่ได้ (พื้นที่ไม่พอ):")
        for it in unplaced:
            print(f"  - {it.id}")

    # สร้างภาพ 3D (ต้องติดตั้ง matplotlib: pip install matplotlib)
    try:
        visualize(container, save_path="/mnt/user-data/outputs/packing_result.png")
    except ImportError:
        print("\n(ข้าม visualization เพราะไม่มี matplotlib ติดตั้งอยู่)")
