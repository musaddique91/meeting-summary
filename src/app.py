import whisper
import os
from pathlib import Path
from transformers import pipeline, AutoTokenizer
import re
import time
import math
from pyannote.audio import Pipeline
from huggingface_hub import login
import moviepy as mv
import summary_action_point as sap

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
action_extractor = pipeline("text2text-generation", model="google/flan-t5-base")


def extract_unique_actions(text, chunk_size=500):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    all_actions = []

    for chunk in chunks:
        prompt = f"""
        Extract UNIQUE action items from this meeting transcript.
        Format as bullets. Skip duplicates. Example:
        - John will send the report
        - Team must review slides

        Transcript: {chunk}
        """
        actions = action_extractor(prompt, max_length=200)[0]['generated_text']
        all_actions.append(actions)

    # Combine and deduplicate
    combined = "\n".join(all_actions)
    unique_actions = list(set([a.strip() for a in combined.split("- ") if a.strip()]))
    return "-\n- ".join([""] + unique_actions)


def save_transcript(file_path, transcript):
    base_path, ext = os.path.splitext(file_path)
    output_path = f"{base_path}_transcript.txt"

    # Write transcript with timestamps
    with open(output_path, "w", encoding="utf-8") as f:
        # Write full text first
        f.write(transcript)
    print(f"Transcript saved to: {output_path}")

def read_transcript(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def transcribe_audio(file_path):
    model = whisper.load_model("base")  # You can also use "small" or "medium"
    result = model.transcribe(file_path)
    save_transcript(file_path, result['text'])
    return result['text'], result['segments']  # Segments contain timestamps


def chunk_text_by_chars(text, max_chars=1024):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            # split very long sentence by max_chars
            for i in range(0, len(sentence), max_chars):
                part = sentence[i:i + max_chars]
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(part.strip())
        else:
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += " " + sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def safe_summarize(text, max_len=130, min_len=30, retries=3):
    for attempt in range(retries):
        try:
            summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Warning: summarization failed on attempt {attempt + 1}/{retries} with error: {e}")
            # Shorten text for retry by cutting last 20%
            cut_len = int(len(text) * 0.8)
            text = text[:cut_len]
            time.sleep(1)  # small delay before retry
    # Final fallback: return first 200 chars with ellipsis
    return text[:200] + " ..."


def summarize_chunks(chunks):
    """Summarize each chunk and return a combined summary."""
    summaries = []

    for idx, chunk in enumerate(chunks):
        print(f"Summarizing chunk {idx + 1}/{len(chunks)} (approx. {len(chunk)} characters)")
        summary = safe_summarize(chunk)
        summaries.append(summary)

    combined = " ".join(summaries)

    # Final summarization pass if still too long
    combined_token_len = len(tokenizer.encode(combined, add_special_tokens=False))
    if combined_token_len > 800:
        print("Running final summarization pass on combined summary...")
        combined = safe_summarize(combined, max_len=180, min_len=60)

    return combined


def summarize_meeting(transcript):
    chunks = chunk_text_by_chars(transcript)
    return summarize_chunks(chunks)

def video_to_audio(path):
    base_name = os.path.splitext(os.path.basename(path))[0]
    directory = os.path.dirname(path)
    output_path = os.path.join(directory, f"{base_name}.mp3")

    # Load video and extract audio
    video = mv.VideoFileClip(path)
    video.audio.write_audiofile(output_path)

    return output_path
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    audio_path = project_root / 'test' / 'meeting-2.mp3'

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
    # transcript = "I'll start again. This is the meeting for the Armour Springfield special meeting to discuss our 2025 financial plan. Today's date is April 29th at 2025 and the time is exactly 6 p.m. I'm Mayor Patrick Terriam for the Armour Springfield. All council is present to my right. The sending order is a Glen Fuel, Andy Kaczynski, Mark Miller and Melinda Warren. If I can get a adoption of the agenda there, move on to seconder please. Melinda and Glen, in addition to council at all. I see none. If I can get a mover or those in support of the adoption of the agenda. Andy, that would be unanimous and is there for or past. We'll get into 4.1 that's adoption of the 2025 financial plan. Can they get a mover and a seconder for that as well please. Melinda and Patrick. I'm going to ask all the council of the Armour Springfield to adopt the 2025 financial plan consisting of one and operating budget to a capital budget. Three and estimate of operating revenue and expenditures for the following fiscal year and four a five year capital expenditure program. Thank you. Questions. Councillor Miller. No, this same. No, we can't hear. I'm sorry. I'm sorry. I'm sorry. I'm sorry. Sorry about that. Just a couple of comments and questions. First one to the CEO or CFO. Under section 3.304 subsection one. No later than May 15th of each year. Thank you. I assume that in the next coming meetings we're going to have a bylaw to adopt the financial plan. Thank you for that clarity. A few concerns I have. I wanted to share with council with this particular proposed 2025 financial plan. I think that's a good question. I think that's a good question. At least one resident got back to me. Actually there's a couple of them a few that had input questions to our council. But as you know, we don't have question period. We don't have delegations and we really have no mechanism for the public to engage with us. But this one individual as you know, it's in our agenda kind of hidden from the public. So I'm just wondering when people submit comments. Because again, we've eliminated question period. We don't really have delegations to this special meeting. So there's no opportunity for the public to engage with us at all. But yet this individual took the effort to do I think a two or three page. Exposition or series of questions for the financial plan. But yet when they submitted them online, this individual had the understanding that they would be posted publicly for people to share his concerns or maybe even respond to them. But I think you know who I'm referring to. I'm not sure if that individual wants me to say that their name. But I'm just wondering when people submit comments. Wasn't that supposed to be the mechanism where the public still has the opportunity to publicly have their concerns shared with the community. In other words, are we just now relying on bank rant and rave for the community to engage with us. This seems to be really the only mechanism. I'm kind of frustrated at that because people are now calling me and saying, Mark, other than you expressing our concerns at these meetings, there's no ability for the public to engage with not only the bureaucracy, but with council in general. And so for some council members, they feel it's gone on deaf ears. In other words, they share those concerns with council representatives, but they don't bring them up. They don't share them and they don't respond to them. And so they feel very frustrated. They feel that the only mechanism now, the only mechanism they have now are two things. One to institute recall legislation, which of course won't happen very quickly. And the second is to vote differently at the next election in October of 2026. So it's unfortunate we've come to this situation where the public really is relegated to the back woods. And so my next point is with the provincial reassessment this year in 2025. We have an addition of $2 million to our kitty. And so we've taken upon us to find ways to spend that $2 million. And you know, I came to council here wanting to sharpen my pencil as much as I could not to cut indiscriminately, but to save taxpayers dollars on discretionary items. In other words, things that are not necessarily for the clear benefit of the ratepayers of our community. And I've tried to do that, but like in any democracy, if they don't flow with the rest of those who vote for those things, they end up in the dungeon and don't get passed. So I would suggest that you know, with this proposed budget, I personally have seen that I'm going to have an 18% increase in my municipal taxes, 18%. And I don't have the biggest house in our municipality. I guarantee you that. My neighbor across the street has a house worth at least five times mine. And they're going to go ballistic if their rate increases 18%. Now, I know that's not the average, but some people are going to have 20% some are going to have 2% some are going to have 8% to they're going to wonder and ask us the question, where are we spending this money? Why is we got $2 million extra because of the reassessment, but yet we're spending that like. I want to be cautious when I say, but we're spending it just because we have it. And so we've got some very important and critical projects coming forward as we all know. And we have to address those. So I would like our council to set a goal of reducing expenditures going forward, not increasing them just because we have a. Our, an influx of $2 million because next year we're not going to have that $2 million extra kitty. And if our expenses keep rising the way they are, you know, people are stretched to the limit. We saw it in the election yesterday affordability and cost of living is the top priority for all political parties, whether they really address it or not. That's another issue, but we have to on the grassroots here in the municipality realize that people don't have an endless credit card to spend money with. I'm not going to go on a rant here, but anyway you kind of get the, the just of where I'm going. Thank you. Thank you, council Miller. I just want to remind people in the audience that there's no recording allowed. I'll be able to understand that. Thank you. Any other questions from council? Council Kazinsky. Yeah, thank you. I would like to point out council a matter that it's not $2 million, but it's $3.2 million. We're going to receive from that 18% increase by reassessment of houses. 3.3, councilor Kazinsky. 3.3. Okay, that even all that even better yet. So I will echo your comments and you concerned that, you know, we have to be really, you know, frugal with our money right now. So because not going to happen all the time that we're going to get that reassessment because province have to, you know, realize that people only have so much money to spend. And the gravy train going to be over shortly. So that's my concern and my input. Thank you. Thank you very much, councilor Kazinsky. Any other questions for come or comments from council? I just like to kind of clarify or summarize everything that as the chair allows me this opportunity to do so. The mechanisms for this budget were totally complied with there. We did have our planning hearing or the public hearing here last week, I believe. There and all the questions were answered. I know who you were talking to and I don't know whether you mentioned his name or not. I'm not sure whether I can or not, but I won't. And those questions were answered by me through on the record there. So they're like, I think they were referred to as flags or something like that. So I answered all those questions there. And it was the questions were answered by or the answers were provided by administration. That weren't answered at the time of the hearing itself there. So I do understand the. The situation that both councils were talking about is Kazinsky Miller and we I totally understand the you know, I don't know that. And our Amazon is a strong position as ourselves. We have very good. Financial position there and it shows in the progress the steps that we're taking, but we don't take it furiously. We went through each one of these as you know, through our committee of a holes and worker groups there. We went detail by detail line by line and saved where we can. And you know, the wants as opposed to needs were all we were all they're all weighed. And so we rely quite heavily on our other directors for their budgets. We give them something at the beginning of the year. And for the everybody does the exceptional job with that with that money and they know it's a taxpayers money. It's most of them are the taxpayers within the armors, make for themselves. So the increase as councilor Miller, it's as stated, they forget what councilor Miller was saying for his increase there, whether you said it or not, I'm not sure. But mine went up 21% of course I'm going to be paying a price for that. My neighbors, we've had discussions about that. And you know, unfortunately, that's out of our control with the assessment year being this year. Yes, we got a pretty good amount of money. We got a pretty good amount of money two years ago. I think we got 2.2 million two years ago. Things change and so on like that. So we we we're able to reduce our excuse me or mill rate by 3.1 per six. So it's not going to help a lot of the taxpayers there and that's on average a 17% increase within the armors, Springfield. Some, you know, might even pay a little bit less there, but it's going as high as 21% in that that would be myself and my neighbors around me there. I do understand the concerns from councilors Miller, Kaczynski there. And just to reiterate there, everybody does have an input into the into this the budget and that was expressed quite eloquently by several people there that came up to us last week. As well as time goes on there, everybody has a chance to discuss this through delegations and questions through their counselors. Myself and I answer every every questionnaire, but that's all I have to say to summarize all that any other questions from council at all. The pretty mayor. Yeah, I just like to say that we did a really good job in working through this budget and we will, like I said, I said it time before that we went through this budget went and reduced the mill rate once went back the second time and even became lower the second time. And again, all administration put their hearts into this to make it work and I agree that all administration and staff that are involved in the budget take it seriously with every resident's dollar and they do the best they can and costs are going up wherever we look. And we know that and we're doing the best we can. And we're going to be able to do a budget to me is a good budget and we got a plan for the future and that's what we're doing because these costs and everything is not going down to keep going up. Thank you. Thank you, Mr. Mayor, it's good we have a council because we have different voice of you. I wouldn't necessarily agree with the deputy mayor in this particular situation. I think we can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. The 9s. The 9s. I think there is improvement to make. I cannot support this budget proposed as it is. I think we can use the excuse that people had an opportunity for a public open house. Thank you, Councillor. Does any other questions? I see none. We can read the resolution. We can vote on it. Yes, please. Here is all the counsel of the arm. Springfield adopted the 2025 financial plan consisting of one and operating budget to a capital budget three and estimate of operating revenue and expenditures for the following fiscal year and for a five year capital expenditure program. With the show hands first of those in support. That will be Councillor Warren, the premier. Fuel and the mayor, Terrin, those opposed. Counselors Miller and Kazinsky, the sole past. Unless there's any other questions from council. I see none. Then proposed to adjourn. Can I move on to second or two adjourn? Councillor Melinda and the mayor and the mayor. Fuel and at 617 p.m. The meeting is adjourned. Thank you very much, people for attending in the audience and online and the council. Thank you very much."
    transcript, segments = transcribe_audio(str(audio_path))
    # print("Transcript:\n", transcript)

    # actions = extract_unique_actions(
    #     "I'll start again. This is the meeting for the Armour Springfield special meeting to discuss our 2025 financial plan. Today's date is April 29th at 2025 and the time is exactly 6 p.m. I'm Mayor Patrick Terriam for the Armour Springfield. All council is present to my right. The sending order is a Glen Fuel, Andy Kaczynski, Mark Miller and Melinda Warren. If I can get a adoption of the agenda there, move on to seconder please. Melinda and Glen, in addition to council at all. I see none. If I can get a mover or those in support of the adoption of the agenda. Andy, that would be unanimous and is there for or past. We'll get into 4.1 that's adoption of the 2025 financial plan. Can they get a mover and a seconder for that as well please. Melinda and Patrick. I'm going to ask all the council of the Armour Springfield to adopt the 2025 financial plan consisting of one and operating budget to a capital budget. Three and estimate of operating revenue and expenditures for the following fiscal year and four a five year capital expenditure program. Thank you. Questions. Councillor Miller. No, this same. No, we can't hear. I'm sorry. I'm sorry. I'm sorry. I'm sorry. Sorry about that. Just a couple of comments and questions. First one to the CEO or CFO. Under section 3.304 subsection one. No later than May 15th of each year. Thank you. I assume that in the next coming meetings we're going to have a bylaw to adopt the financial plan. Thank you for that clarity. A few concerns I have. I wanted to share with council with this particular proposed 2025 financial plan. I think that's a good question. I think that's a good question. At least one resident got back to me. Actually there's a couple of them a few that had input questions to our council. But as you know, we don't have question period. We don't have delegations and we really have no mechanism for the public to engage with us. But this one individual as you know, it's in our agenda kind of hidden from the public. So I'm just wondering when people submit comments. Because again, we've eliminated question period. We don't really have delegations to this special meeting. So there's no opportunity for the public to engage with us at all. But yet this individual took the effort to do I think a two or three page. Exposition or series of questions for the financial plan. But yet when they submitted them online, this individual had the understanding that they would be posted publicly for people to share his concerns or maybe even respond to them. But I think you know who I'm referring to. I'm not sure if that individual wants me to say that their name. But I'm just wondering when people submit comments. Wasn't that supposed to be the mechanism where the public still has the opportunity to publicly have their concerns shared with the community. In other words, are we just now relying on bank rant and rave for the community to engage with us. This seems to be really the only mechanism. I'm kind of frustrated at that because people are now calling me and saying, Mark, other than you expressing our concerns at these meetings, there's no ability for the public to engage with not only the bureaucracy, but with council in general. And so for some council members, they feel it's gone on deaf ears. In other words, they share those concerns with council representatives, but they don't bring them up. They don't share them and they don't respond to them. And so they feel very frustrated. They feel that the only mechanism now, the only mechanism they have now are two things. One to institute recall legislation, which of course won't happen very quickly. And the second is to vote differently at the next election in October of 2026. So it's unfortunate we've come to this situation where the public really is relegated to the back woods. And so my next point is with the provincial reassessment this year in 2025. We have an addition of $2 million to our kitty. And so we've taken upon us to find ways to spend that $2 million. And you know, I came to council here wanting to sharpen my pencil as much as I could not to cut indiscriminately, but to save taxpayers dollars on discretionary items. In other words, things that are not necessarily for the clear benefit of the ratepayers of our community. And I've tried to do that, but like in any democracy, if they don't flow with the rest of those who vote for those things, they end up in the dungeon and don't get passed. So I would suggest that you know, with this proposed budget, I personally have seen that I'm going to have an 18% increase in my municipal taxes, 18%. And I don't have the biggest house in our municipality. I guarantee you that. My neighbor across the street has a house worth at least five times mine. And they're going to go ballistic if their rate increases 18%. Now, I know that's not the average, but some people are going to have 20% some are going to have 2% some are going to have 8% to they're going to wonder and ask us the question, where are we spending this money? Why is we got $2 million extra because of the reassessment, but yet we're spending that like. I want to be cautious when I say, but we're spending it just because we have it. And so we've got some very important and critical projects coming forward as we all know. And we have to address those. So I would like our council to set a goal of reducing expenditures going forward, not increasing them just because we have a. Our, an influx of $2 million because next year we're not going to have that $2 million extra kitty. And if our expenses keep rising the way they are, you know, people are stretched to the limit. We saw it in the election yesterday affordability and cost of living is the top priority for all political parties, whether they really address it or not. That's another issue, but we have to on the grassroots here in the municipality realize that people don't have an endless credit card to spend money with. I'm not going to go on a rant here, but anyway you kind of get the, the just of where I'm going. Thank you. Thank you, council Miller. I just want to remind people in the audience that there's no recording allowed. I'll be able to understand that. Thank you. Any other questions from council? Council Kazinsky. Yeah, thank you. I would like to point out council a matter that it's not $2 million, but it's $3.2 million. We're going to receive from that 18% increase by reassessment of houses. 3.3, councilor Kazinsky. 3.3. Okay, that even all that even better yet. So I will echo your comments and you concerned that, you know, we have to be really, you know, frugal with our money right now. So because not going to happen all the time that we're going to get that reassessment because province have to, you know, realize that people only have so much money to spend. And the gravy train going to be over shortly. So that's my concern and my input. Thank you. Thank you very much, councilor Kazinsky. Any other questions for come or comments from council? I just like to kind of clarify or summarize everything that as the chair allows me this opportunity to do so. The mechanisms for this budget were totally complied with there. We did have our planning hearing or the public hearing here last week, I believe. There and all the questions were answered. I know who you were talking to and I don't know whether you mentioned his name or not. I'm not sure whether I can or not, but I won't. And those questions were answered by me through on the record there. So they're like, I think they were referred to as flags or something like that. So I answered all those questions there. And it was the questions were answered by or the answers were provided by administration. That weren't answered at the time of the hearing itself there. So I do understand the. The situation that both councils were talking about is Kazinsky Miller and we I totally understand the you know, I don't know that. And our Amazon is a strong position as ourselves. We have very good. Financial position there and it shows in the progress the steps that we're taking, but we don't take it furiously. We went through each one of these as you know, through our committee of a holes and worker groups there. We went detail by detail line by line and saved where we can. And you know, the wants as opposed to needs were all we were all they're all weighed. And so we rely quite heavily on our other directors for their budgets. We give them something at the beginning of the year. And for the everybody does the exceptional job with that with that money and they know it's a taxpayers money. It's most of them are the taxpayers within the armors, make for themselves. So the increase as councilor Miller, it's as stated, they forget what councilor Miller was saying for his increase there, whether you said it or not, I'm not sure. But mine went up 21% of course I'm going to be paying a price for that. My neighbors, we've had discussions about that. And you know, unfortunately, that's out of our control with the assessment year being this year. Yes, we got a pretty good amount of money. We got a pretty good amount of money two years ago. I think we got 2.2 million two years ago. Things change and so on like that. So we we we're able to reduce our excuse me or mill rate by 3.1 per six. So it's not going to help a lot of the taxpayers there and that's on average a 17% increase within the armors, Springfield. Some, you know, might even pay a little bit less there, but it's going as high as 21% in that that would be myself and my neighbors around me there. I do understand the concerns from councilors Miller, Kaczynski there. And just to reiterate there, everybody does have an input into the into this the budget and that was expressed quite eloquently by several people there that came up to us last week. As well as time goes on there, everybody has a chance to discuss this through delegations and questions through their counselors. Myself and I answer every every questionnaire, but that's all I have to say to summarize all that any other questions from council at all. The pretty mayor. Yeah, I just like to say that we did a really good job in working through this budget and we will, like I said, I said it time before that we went through this budget went and reduced the mill rate once went back the second time and even became lower the second time. And again, all administration put their hearts into this to make it work and I agree that all administration and staff that are involved in the budget take it seriously with every resident's dollar and they do the best they can and costs are going up wherever we look. And we know that and we're doing the best we can. And we're going to be able to do a budget to me is a good budget and we got a plan for the future and that's what we're doing because these costs and everything is not going down to keep going up. Thank you. Thank you, Mr. Mayor, it's good we have a council because we have different voice of you. I wouldn't necessarily agree with the deputy mayor in this particular situation. I think we can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. We can do better. The 9s. The 9s. I think there is improvement to make. I cannot support this budget proposed as it is. I think we can use the excuse that people had an opportunity for a public open house. Thank you, Councillor. Does any other questions? I see none. We can read the resolution. We can vote on it. Yes, please. Here is all the counsel of the arm. Springfield adopted the 2025 financial plan consisting of one and operating budget to a capital budget three and estimate of operating revenue and expenditures for the following fiscal year and for a five year capital expenditure program. With the show hands first of those in support. That will be Councillor Warren, the premier. Fuel and the mayor, Terrin, those opposed. Counselors Miller and Kazinsky, the sole past. Unless there's any other questions from council. I see none. Then proposed to adjourn. Can I move on to second or two adjourn? Councillor Melinda and the mayor and the mayor. Fuel and at 617 p.m. The meeting is adjourned. Thank you very much, people for attending in the audience and online and the council. Thank you very much.")
    # print("Extracted Actions:\n", actions)
    # summary = summarize_meeting(
    #     )
    # print("Summary:\n", summary)
    # print("Segments:\n", segments)
    summary_actions = sap.get_action_items(transcript)
    print(summary_actions)
