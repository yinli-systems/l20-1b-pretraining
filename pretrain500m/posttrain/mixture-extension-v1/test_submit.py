from submit import runnable_four_gpu_nodes


def test_parses_public_four_gpu_layout_from_capacity_dashboard():
    text = """public partition:\nPartition free total\ngpu_5090 42 248\n\nrunnable:\nPartition 1 2 3 4 5 6 7 8\ngpu_5090 42 16 9 4 3 2 1 0\n"""
    assert runnable_four_gpu_nodes(text) == 4
