import pytest
import amendements_analysis.settings.base as stg
from amendements_analysis.infrastructure.sentence_encoding import TextEncoder


@pytest.mark.parametrize('sentence, finetuned, batch_size, expected', 
                       [(['La santé avant tout'], True, 1, (1, 768)),
                        (['La santé avant tout'], False, 1, (1,768)),
                        (['La santé avant tout','Le travail avant tout'], True, 1, (2,768))])                                              
def test_sentence_embeddings(sentence, finetuned, batch_size, expected):
    assert TextEncoder(sentence, finetuned, batch_size).sentence_embeddings.shape == expected

