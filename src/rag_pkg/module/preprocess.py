import pandas as pd

def filter_valid_reviews(df: pd.DataFrame, min_comments: int = 2) -> pd.DataFrame:
    """
    RAG에 적합한 유효한 위스키 리뷰만 필터링합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        원본 위스키 리뷰 데이터프레임
    min_comments : int, default=2
        최소 필요한 Comment 개수 (Nose/Taste/Finish 중)
    
    Returns:
    --------
    pd.DataFrame
        필터링된 데이터프레임

    """
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
    """
    각 위스키 리뷰를 RAG용 단일 문서 텍스트로 결합합니다.

    Parameters:
    -----------
    row : pd.Series
        위스키 리뷰 데이터의 한 행

    Returns:
    --------
    str
        임베딩에 사용할 구조화된 텍스트

    Format:
    -------
    위스키 이름: {name}
    태그: {tags}

    향(Nose): {nose_comment}

    맛(Taste): {taste_comment}

    피니쉬(Finish): {finish_comment}
    """
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
    """
    위스키 리뷰 데이터를 RAG 시스템에 사용하기 위해 전처리합니다.

    이 함수는 다음 단계를 수행합니다:
    1. 유효한 리뷰만 필터링 (filter_valid_reviews)
    2. RAG용 문서 텍스트 생성 (create_document_text)

    Parameters:
    -----------
    df : pd.DataFrame
        원본 위스키 리뷰 데이터프레임
    min_comments : int, default=2
        최소 필요한 Comment 개수 (Nose/Taste/Finish 중)
    add_document_text : bool, default=True
        document_text 컬럼 추가 여부

    Returns:
    --------
    pd.DataFrame
        전처리된 데이터프레임 (document_text 컬럼 포함)
    """
    # 1. 유효한 리뷰만 필터링
    filtered_df = filter_valid_reviews(df, min_comments=min_comments)

    # 2. RAG용 문서 텍스트 생성
    if add_document_text:
        filtered_df['document_text'] = filtered_df.apply(create_document_text, axis=1)

    return filtered_df