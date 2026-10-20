from pal.context_manager.components.pager import ContextPager, Segment


def test_page_out_splits_hot_cold():
    pager = ContextPager(hot_window=2)
    history = [Segment(id=str(i), content=f"msg {i}") for i in range(5)]
    hot, cold_refs = pager.page_out(history)
    assert len(hot) == 2
    assert len(cold_refs) == 3


def test_page_out_no_split_when_short():
    pager = ContextPager(hot_window=6)
    history = [Segment(id=str(i), content=f"msg {i}") for i in range(3)]
    hot, cold_refs = pager.page_out(history)
    assert len(hot) == 3
    assert cold_refs == []
