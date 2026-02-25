"""Bag topic probing helpers."""

import rosbag


def find_actual_hand_state_topic(bag_file: str, hand_state_topics: list[str]) -> str | None:
    bag = rosbag.Bag(bag_file)
    try:
        bag_topics = set([t for t in bag.get_type_and_topic_info().topics])
    finally:
        bag.close()
    for t in hand_state_topics:
        if t in bag_topics:
            return t
    return None


def test_joint_current_availability(bag_file: str, sensors_topic: str) -> bool:
    """
    True: 具备 joint_current
    False: 不具备 joint_current（可能只有 joint_torque）
    """
    try:
        bag = rosbag.Bag(bag_file)
        try:
            for _, msg, _ in bag.read_messages(topics=[sensors_topic]):
                try:
                    _ = msg.joint_data.joint_current
                    return True
                except AttributeError:
                    try:
                        _ = msg.joint_data.joint_torque
                        return False
                    except AttributeError:
                        return False
                break
            return False
        finally:
            bag.close()
    except Exception:
        return False

