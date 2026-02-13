"""
Level 0 检查：LeRobot 数据集格式验证
- 验证数据集能否被 LeRobot 库正确加载
- 验证数据集的完整性和可访问性
- 一票否决：如果格式检查失败，后续检查将不会执行

检查内容：
1. 检查总帧数是否一致
2. 检查总 episodes 数是否一致
3. 抽样前中后三个点，看看是否能导入内存
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class LeRobotFormatValidator:
    """LeRobot 数据集格式验证器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化验证器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
    def validate(self, dataset_path: Path) -> Dict[str, Any]:
        """
        验证数据集是否符合 LeRobot 格式
        
        检查内容：
        1. 检查总帧数是否一致
        2. 检查总 episodes 数是否一致
        3. 抽样前中后三个点，看看是否能导入内存
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            验证结果字典
        """
        logger.info("=" * 60)
        logger.info("Level 0: LeRobot 格式验证")
        logger.info("=" * 60)
        
        result = {
            'passed': False,
            'can_load': False,
            'total_frames': 0,
            'fps': 0,
            'total_episodes': 0,
            'errors': [],
            'warnings': []
        }
        
        try:
            # 尝试导入 LeRobot
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except ImportError:
                try:
                    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
                except ImportError as e:
                    error_msg = f"无法导入 LeRobot 库: {str(e)}"
                    logger.error(error_msg)
                    result['errors'].append(error_msg)
                    result['errors'].append("请确保已安装 lerobot 包: pip install lerobot")
                    logger.error("=" * 60)
                    logger.error("❌ Level 0 检查失败：无法导入 LeRobot 库")
                    logger.error("=" * 60)
                    return result
            
            # 尝试加载数据集
            logger.info(f"尝试加载数据集: {dataset_path}")
            
            try:
                # 加载数据集
                dataset = LeRobotDataset(
                    repo_id=str(dataset_path),
                    image_transforms=None,
                    delta_timestamps=None,
                    video_backend='torchcodec'
                )
                
                logger.info("✅ 数据集加载成功")
                result['can_load'] = True
                
            except Exception as e:
                error_msg = f"加载数据集失败: {str(e)}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
                
                # 提供更具体的错误信息
                if "No such file or directory" in str(e):
                    result['errors'].append("数据集路径不存在或不可访问")
                elif "meta/info.json" in str(e):
                    result['errors'].append("缺少 meta/info.json 文件")
                elif "data" in str(e):
                    result['errors'].append("数据文件损坏或格式不正确")
                
                logger.error("=" * 60)
                logger.error("❌ Level 0 检查失败：数据集加载失败")
                logger.error("=" * 60)
                return result
            
            # 检查 1: 总帧数是否一致
            try:
                total_frames = len(dataset)
                expected_frames = dataset.meta.info.get('total_frames', 0)
                
                result['total_frames'] = total_frames
                result['fps'] = dataset.meta.fps
                
                logger.info(f"检查总帧数: 实际={total_frames}, 预期={expected_frames}")
                
                if total_frames != expected_frames:
                    error_msg = f"总帧数不一致: 实际={total_frames}, 预期={expected_frames}"
                    logger.error(error_msg)
                    result['errors'].append(error_msg)
                    logger.error("=" * 60)
                    logger.error("❌ Level 0 检查失败：总帧数不一致")
                    logger.error("=" * 60)
                    return result
                
                logger.info("✅ 总帧数一致")
                
            except Exception as e:
                error_msg = f"检查总帧数失败: {str(e)}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
                logger.error("=" * 60)
                logger.error("❌ Level 0 检查失败：无法获取帧数信息")
                logger.error("=" * 60)
                return result
            
            # 检查 2: 总 episodes 数是否一致
            try:
                actual_episodes = len(dataset.episode_data_index['from'])
                expected_episodes = dataset.meta.info.get('total_episodes', 0)
                
                result['total_episodes'] = actual_episodes
                
                logger.info(f"检查总 episodes 数: 实际={actual_episodes}, 预期={expected_episodes}")
                
                if actual_episodes != expected_episodes:
                    error_msg = f"总 episodes 数不一致: 实际={actual_episodes}, 预期={expected_episodes}"
                    logger.error(error_msg)
                    result['errors'].append(error_msg)
                    logger.error("=" * 60)
                    logger.error("❌ Level 0 检查失败：总 episodes 数不一致")
                    logger.error("=" * 60)
                    return result
                
                logger.info("✅ 总 episodes 数一致")
                
            except Exception as e:
                error_msg = f"检查总 episodes 数失败: {str(e)}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
                logger.error("=" * 60)
                logger.error("❌ Level 0 检查失败：无法获取 episodes 信息")
                logger.error("=" * 60)
                return result
            
            # 检查 3: 抽样前中后三个点，看看是否能导入内存
            try:
                logger.info("抽样检查数据可访问性（前、中、后三个点）...")
                
                # 前、中、后三个采样点
                sample_indices = [0, (total_frames - 1) // 2, total_frames - 1]
                
                for i in sample_indices:
                    try:
                        tmp = dataset[i]
                        del tmp
                        logger.info(f"  ✅ 帧 {i} 可访问")
                    except Exception as e:
                        error_msg = f"帧 {i} 无法访问: {str(e)}"
                        logger.error(f"  ❌ {error_msg}")
                        result['errors'].append(error_msg)
                        logger.error("=" * 60)
                        logger.error("❌ Level 0 检查失败：数据无法访问")
                        logger.error("=" * 60)
                        return result
                
                logger.info("✅ 抽样检查通过（前、中、后三个点均可访问）")
                
            except Exception as e:
                error_msg = f"抽样检查失败: {str(e)}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
                logger.error("=" * 60)
                logger.error("❌ Level 0 检查失败：抽样检查失败")
                logger.error("=" * 60)
                return result
            
            # 所有检查通过
            result['passed'] = True
            logger.info("=" * 60)
            logger.info("✅ Level 0 检查通过：数据集符合 LeRobot 格式")
            logger.info(f"   - 总帧数: {total_frames}")
            logger.info(f"   - 总 Episodes: {actual_episodes}")
            logger.info(f"   - FPS: {result['fps']}")
            logger.info("=" * 60)
            
        except Exception as e:
            error_msg = f"验证过程发生未预期的错误: {str(e)}"
            logger.error(error_msg)
            result['errors'].append(error_msg)
            import traceback
            logger.debug(traceback.format_exc())
            logger.error("=" * 60)
            logger.error("❌ Level 0 检查失败")
            logger.error("=" * 60)
        
        return result
    
    def validate_full(self, dataset_path: Path) -> Dict[str, Any]:
        """
        完整验证（访问所有帧，较慢但更彻底）
        
        注意：此方法会访问所有帧，速度较慢，仅用于发布前的最终检查
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            验证结果字典
        """
        # 先执行基本检查
        result = self.validate(dataset_path)
        
        if not result['passed']:
            return result
        
        # 如果基本检查通过，继续完整检查
        logger.info("=" * 60)
        logger.info("开始完整验证（访问所有帧）...")
        logger.info("=" * 60)
        
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            
            dataset = LeRobotDataset(
                repo_id=str(dataset_path),
                image_transforms=None,
                delta_timestamps=None,
                video_backend='torchcodec'
            )
            
            total_frames = len(dataset)
            
            for i in range(total_frames):
                try:
                    tmp = dataset[i]
                    del tmp
                except Exception as e:
                    error_msg = f"帧 {i} 无法访问: {str(e)}"
                    logger.error(error_msg)
                    result['errors'].append(error_msg)
                    result['passed'] = False
                    logger.error("=" * 60)
                    logger.error("❌ 完整验证失败")
                    logger.error("=" * 60)
                    return result

                # 每1000帧输出一次进度
                if (i + 1) % 1000 == 0:
                    logger.info(f"已验证 {i + 1}/{total_frames} 帧 ({(i+1)/total_frames*100:.1f}%)")

            logger.info("=" * 60)
            logger.info(f"✅ 完整验证通过（访问了所有 {total_frames} 帧）")
            logger.info("=" * 60)

        except Exception as e:
            error_msg = f"完整验证失败: {str(e)}"
            logger.error(error_msg)
            result['errors'].append(error_msg)
            result['passed'] = False
            logger.error("=" * 60)
            logger.error("❌ 完整验证失败")
            logger.error("=" * 60)

        return result

