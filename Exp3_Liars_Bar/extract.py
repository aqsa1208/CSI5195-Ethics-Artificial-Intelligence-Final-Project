import os
import json
import pandas as pd
from collections import defaultdict

json_folder_path = "C:\\Users\\Gawai\\Desktop\\Liars_Bar"
data = []

for filename in os.listdir(json_folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(json_folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                game = json.load(f)
            except json.JSONDecodeError as e:
                print(f"跳过无效文件: {filename} -> {e}")
                continue

        game_id = game["game_id"]
        player_names = game["player_names"]


        last_round_id = None
        self_shoot_counts = defaultdict(int)  # 累积对自己开枪次数
        
        
        for round_data in game["rounds"]:
            round_id = round_data["round_id"]
            target_card = round_data["target_card"]
            round_players = round_data["round_players"]
            play_history = round_data["play_history"]


            # 如果 round_id 从较大数字跳回到1，说明新一局开始，需要重置开枪记录
            if last_round_id is not None and round_id < last_round_id:
                self_shoot_counts = defaultdict(int)  # 清零累计
            last_round_id = round_id  # 更新 last_round_id


            # 记录每个玩家是否曾成功质疑（用于历史信息）
            successful_challenges = defaultdict(bool)

            for i, play in enumerate(play_history):
                player_name = play["player_name"]
                played_cards = play["played_cards"]
                was_challenged = play.get("was_challenged", False)
                challenge_result = play.get("challenge_result", None)
                next_player = play.get("next_player", None)

                # 累计到当前为止，这轮一共打出了多少张目标牌
                total_claimed_target_card = sum(
                    p["played_cards"].count(target_card)
                    for p in play_history[:i + 1]
                )

                # challenger：无论是否真的质疑，都记录是谁 *本来可以* 质疑他
                challenger = next_player
                
                
                # ===== 统计谁“对自己开了枪” =====
                if was_challenged:
                    if challenge_result is True:
                    # 被质疑成功，出牌者开枪
                        self_shoot_counts[player_name] += 1
                    elif challenge_result is False and challenger:
                        # 质疑失败，质疑者开枪
                        self_shoot_counts[challenger] += 1

                # 上一家玩家是谁（上一条记录的出牌者）
                previous_player = play_history[i]["player_name"]
                previous_self_shots = self_shoot_counts[previous_player] if previous_player else 0
                
                # 当前玩家累计开枪数
                own_self_shots = self_shoot_counts[next_player]


                # 如果曾成功质疑，标记下来
                if was_challenged and challenge_result and challenger:
                    successful_challenges[challenger] = True

                # 当前这个 challenger 曾成功质疑过吗？
                previous_successful_challenge = successful_challenges.get(challenger, False)

                # 获取 challenger 的手牌中目标牌数量 & joker 数量
                target_card_count_in_hand = None
                joker_count = None
                if challenger:
                    for state in round_data["player_initial_states"]:
                        if state["player_name"] == challenger:
                            target_card_count_in_hand = state["initial_hand"].count(target_card)
                            joker_count = state["initial_hand"].count("Joker")
                            break

                # 提取挑战者 opinion
                opinion_text = None
                if challenger and round_data.get("player_opinions"):
                    challenger_opinions = round_data["player_opinions"].get(challenger, {})
                    opinion_text = challenger_opinions.get(player_name, None)

                # 行为和文本
                behavior = play.get("behavior", None)
                challenge_thinking = play.get("challenge_thinking", "")
                play_thinking = play.get("play_thinking", "")

                data.append({
                    "game_id": game_id,
                    "round_id": round_id,
                    "player": player_name,
                    "was_challenged": was_challenged,
                    "challenger": challenger,
                    "challenge_result": challenge_result,
                    "total_claimed_target_card": total_claimed_target_card,
                    "target_card_count_in_hand": target_card_count_in_hand,
                    "joker_count": joker_count,
                    "previous_successful_challenge": previous_successful_challenge,
                    "opinion_text": opinion_text,
                    "behavior": behavior,
                    "current_play_index": i + 1,
                    "target_card": target_card,
                    "previous_self_shots": previous_self_shots,
                    "Challenger_self_shots": own_self_shots
                })

# 保存为 CSV 文件
current_script_dir = os.path.dirname(os.path.abspath(__file__))
output_csv_path = os.path.join(current_script_dir, "all_plays_with_challenge_info_1.csv")

df = pd.DataFrame(data)
df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"✅ 提取完毕，已保存为 {output_csv_path}")
