import os

# 把这里改成你的图片所在文件夹
output_dir = 'C:/Users/G/Downloads/Processed_Widar_6AP'

# 根据用户 ID 映射对应的房间 (环境)
room_mapping = {
    # Room 1 (1130包)
    'U05': 'E1', 'U10': 'E1', 'U11': 'E1', 'U12': 'E1', 
    'U13': 'E1', 'U14': 'E1', 'U15': 'E1', 'U16': 'E1', 'U17': 'E1',
    # Room 2 (1204~1209包，已删除U02和U03)
    'U01': 'E2', 'U06': 'E2',
    # Room 3 (1211包，已删除U03)
    'U07': 'E3', 'U08': 'E3', 'U09': 'E3'
}

count = 0
for filename in os.listdir(output_dir):
    if filename.endswith(".png") and not filename.startswith("E"):
        # 提取前面的 Uxx
        user_prefix = filename[:3] 
        
        if user_prefix in room_mapping:
            env_prefix = room_mapping[user_prefix]
            new_name = f"{env_prefix}_{filename}"
            
            old_path = os.path.join(output_dir, filename)
            new_path = os.path.join(output_dir, new_name)
            
            os.rename(old_path, new_path)
            count += 1

print(f"✅ 成功抢救并重命名了 {count} 张图片！")