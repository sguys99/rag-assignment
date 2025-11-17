import pandas as pd

def filter_valid_reviews(df: pd.DataFrame, min_comments: int = 2) -> pd.DataFrame:
    df_filtered = df.copy()
    
    essential_mask = (
        df_filtered['Whisky Name'].notna() & 
        (df_filtered['Whisky Name'].str.strip() != '') &
        df_filtered['Link'].notna() & 
        (df_filtered['Link'].str.strip() != '')
    )
    
    comment_cols = ['Nose Comment', 'Taste Comment', 'Finish Comment']
    
    valid_comments_count = 0
    for col in comment_cols:
        valid_comments_count += (
            df_filtered[col].notna() & 
            (df_filtered[col].str.strip() != '')
        ).astype(int)
    
    comments_mask = valid_comments_count >= min_comments
    final_mask = essential_mask & comments_mask
    df_result = df_filtered[final_mask].copy()
    df_result = df_result.reset_index(drop=True)
    
    return df_result


def create_document_text(row: pd.Series) -> str:
    parts = [f"위스키 이름: {row['Whisky Name']}"]

    if pd.notna(row['Tags']) and row['Tags'].strip():
        parts.append(f"태그: {row['Tags']}")

    parts.append("")

    if pd.notna(row['Nose Comment']) and row['Nose Comment'].strip():
        nose_text = f"향(Nose)"
        if pd.notna(row['Nose Score']) and str(row['Nose Score']).strip():
            nose_text += f" [점수: {row['Nose Score']}]"
        nose_text += f": {row['Nose Comment']}"
        parts.append(nose_text)
        parts.append("")

    # Taste Comment
    if pd.notna(row['Taste Comment']) and row['Taste Comment'].strip():
        taste_text = f"맛(Taste)"
        if pd.notna(row['Taste Score']) and str(row['Taste Score']).strip():
            taste_text += f" [점수: {row['Taste Score']}]"
        taste_text += f": {row['Taste Comment']}"
        parts.append(taste_text)
        parts.append("")

    # Finish Comment
    if pd.notna(row['Finish Comment']) and row['Finish Comment'].strip():
        finish_text = f"피니쉬(Finish)"
        if pd.notna(row['Finish Score']) and str(row['Finish Score']).strip():
            finish_text += f" [점수: {row['Finish Score']}]"
        finish_text += f": {row['Finish Comment']}"
        parts.append(finish_text)

    return "\n".join(parts)


def preprocess_for_rag(
    df: pd.DataFrame,
    min_comments: int = 2,
    add_document_text: bool = True
) -> pd.DataFrame:
    # 1. 유효한 리뷰만 필터링
    filtered_df = filter_valid_reviews(df, min_comments=min_comments)

    # 2. RAG용 문서 텍스트 생성
    if add_document_text:
        filtered_df['document_text'] = filtered_df.apply(create_document_text, axis=1)

    return filtered_df