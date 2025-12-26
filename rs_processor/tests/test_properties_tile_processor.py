"""
分块处理属性测试
"""

import pytest
from hypothesis import given, strategies as st, settings

from rs_processor.processing.tile_processor import TileProcessor, TileInfo


# **Feature: image-processing-refactor, Property 17: 分块覆盖完整性**
@given(
    img_width=st.integers(min_value=1000, max_value=20000),
    img_height=st.integers(min_value=1000, max_value=20000),
    tile_size=st.integers(min_value=2000, max_value=10000),
    overlap=st.integers(min_value=0, max_value=400),
)
@settings(max_examples=100, deadline=None)
def test_tiles_cover_entire_image(img_width, img_height, tile_size, overlap):
    """Property 17: 分块覆盖完整性"""
    if overlap >= tile_size // 2:
        overlap = tile_size // 4
    
    processor = TileProcessor(tile_size=tile_size, overlap=overlap)
    tiles = processor.generate_tiles(img_width, img_height)
    
    assert len(tiles) > 0
    
    test_points = [(0, 0), (img_width - 1, 0), (0, img_height - 1), (img_width - 1, img_height - 1), (img_width // 2, img_height // 2)]
    for px, py in test_points:
        covered = any(tile.x <= px < tile.x + tile.width and tile.y <= py < tile.y + tile.height for tile in tiles)
        assert covered, f"点 ({px}, {py}) 未被任何分块覆盖"


@given(img_width=st.integers(min_value=2000, max_value=20000), img_height=st.integers(min_value=2000, max_value=20000))
@settings(max_examples=50)
def test_tiles_have_valid_bounds(img_width, img_height):
    """所有分块的边界应该在影像范围内"""
    processor = TileProcessor(tile_size=8000, overlap=200)
    tiles = processor.generate_tiles(img_width, img_height)
    
    for tile in tiles:
        assert tile.x >= 0 and tile.y >= 0
        assert tile.x + tile.width <= img_width
        assert tile.y + tile.height <= img_height
        assert tile.width > 0 and tile.height > 0


# **Feature: image-processing-refactor, Property 18: 标注完整性保护**
@given(
    ann_x=st.integers(min_value=100, max_value=5000),
    ann_y=st.integers(min_value=100, max_value=5000),
    ann_w=st.integers(min_value=50, max_value=500),
    ann_h=st.integers(min_value=50, max_value=500),
)
@settings(max_examples=100)
def test_annotation_contained_in_at_least_one_tile(ann_x, ann_y, ann_w, ann_h):
    """Property 18: 标注完整性保护"""
    img_width, img_height = 10000, 10000
    ann_x = min(ann_x, img_width - ann_w - 1)
    ann_y = min(ann_y, img_height - ann_h - 1)
    
    annotations = [{'bbox': (ann_x, ann_y, ann_w, ann_h)}]
    processor = TileProcessor(tile_size=4000, overlap=200)
    tiles = processor.generate_tiles(img_width, img_height, annotations)
    
    assert any(0 in tile.contains_annotations for tile in tiles)


@given(num_annotations=st.integers(min_value=1, max_value=10))
@settings(max_examples=50)
def test_filter_annotations_returns_relative_coords(num_annotations):
    """过滤后的标注应该使用相对坐标"""
    annotations = [{'bbox': (1000 + i * 500, 1000 + i * 500, 200, 200), 'id': i} for i in range(num_annotations)]
    processor = TileProcessor(tile_size=8000, overlap=200)
    tiles = processor.generate_tiles(10000, 10000, annotations)
    
    if tiles:
        tile = tiles[0]
        filtered = processor.filter_annotations_in_tile(annotations, tile)
        for ann in filtered:
            if tile.x > 0 or tile.y > 0:
                original_ann = next(a for a in annotations if a.get('id') == ann.get('id'))
                orig_x, orig_y, _, _ = original_ann['bbox']
                rel_x, rel_y, _, _ = ann['bbox']
                assert rel_x == orig_x - tile.x
                assert rel_y == orig_y - tile.y


def test_tile_info_to_dict():
    """TileInfo 序列化应该正确"""
    tile = TileInfo(x=1000, y=2000, width=8000, height=8000, tile_idx=5, contains_annotations=[0, 2, 5])
    d = tile.to_dict()
    assert d['x'] == 1000 and d['y'] == 2000
    assert d['width'] == 8000 and d['height'] == 8000
    assert d['tile_idx'] == 5
    assert d['contains_annotations'] == [0, 2, 5]


@given(tile_size=st.integers(min_value=1000, max_value=10000), overlap=st.integers(min_value=0, max_value=500))
@settings(max_examples=50)
def test_tiles_have_sequential_indices(tile_size, overlap):
    """分块索引应该是连续的"""
    if overlap >= tile_size:
        overlap = tile_size // 4
    processor = TileProcessor(tile_size=tile_size, overlap=overlap)
    tiles = processor.generate_tiles(15000, 15000)
    assert [t.tile_idx for t in tiles] == list(range(len(tiles)))
