use ggml_all_sys_2::{
    WHISPER_GRETYPE_ALT, WHISPER_GRETYPE_CHAR,
    WHISPER_GRETYPE_CHAR_ALT, WHISPER_GRETYPE_CHAR_NOT,
    WHISPER_GRETYPE_CHAR_RNG_UPPER, WHISPER_GRETYPE_END,
    WHISPER_GRETYPE_RULE_REF,
};

#[cfg_attr(any(not(windows), target_env = "gnu"), repr(u32))] // include windows-gnu
#[cfg_attr(all(windows, not(target_env = "gnu")), repr(i32))] // msvc being *special* again
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum WhisperGrammarElementType {
    /// End of rule definition
    End = WHISPER_GRETYPE_END,
    /// Start of alternate definition for a rule
    Alternate = WHISPER_GRETYPE_ALT,
    /// Non-terminal element: reference to another rule
    RuleReference = WHISPER_GRETYPE_RULE_REF,
    /// Terminal element: character (code point)
    Character = WHISPER_GRETYPE_CHAR,
    /// Inverse of a character(s)
    NotCharacter = WHISPER_GRETYPE_CHAR_NOT,
    /// Modifies a preceding [Self::Character] to be an inclusive range
    CharacterRangeUpper = WHISPER_GRETYPE_CHAR_RNG_UPPER,
    /// Modifies a preceding [Self::Character] to add an alternate character to match
    CharacterAlternate = WHISPER_GRETYPE_CHAR_ALT,
}

impl From<ggml_all_sys_2::whisper_gretype> for WhisperGrammarElementType {
    fn from(value: ggml_all_sys_2::whisper_gretype) -> Self {
        assert!(
            (0..=6).contains(&value),
            "Invalid WhisperGrammarElementType value: {}",
            value
        );

        #[allow(non_upper_case_globals)] // weird place to trigger this
        match value {
            WHISPER_GRETYPE_END => WhisperGrammarElementType::End,
            WHISPER_GRETYPE_ALT => WhisperGrammarElementType::Alternate,
            WHISPER_GRETYPE_RULE_REF => WhisperGrammarElementType::RuleReference,
            WHISPER_GRETYPE_CHAR => WhisperGrammarElementType::Character,
            WHISPER_GRETYPE_CHAR_NOT => WhisperGrammarElementType::NotCharacter,
            WHISPER_GRETYPE_CHAR_RNG_UPPER => {
                WhisperGrammarElementType::CharacterRangeUpper
            }
            WHISPER_GRETYPE_CHAR_ALT => {
                WhisperGrammarElementType::CharacterAlternate
            }
            _ => unreachable!(),
        }
    }
}

impl From<WhisperGrammarElementType> for ggml_all_sys_2::whisper_gretype {
    fn from(value: WhisperGrammarElementType) -> Self {
        value as Self
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct WhisperGrammarElement {
    pub element_type: WhisperGrammarElementType,
    pub value: u32,
}

impl WhisperGrammarElement {
    pub fn new(element_type: WhisperGrammarElementType, value: u32) -> Self {
        Self {
            element_type,
            value,
        }
    }

    pub fn to_c_type(self) -> ggml_all_sys_2::whisper_grammar_element {
        ggml_all_sys_2::whisper_grammar_element {
            type_: self.element_type.into(),
            value: self.value,
        }
    }
}
