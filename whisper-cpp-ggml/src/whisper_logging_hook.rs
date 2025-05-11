use crate::common_logging::{
    generic_debug, generic_error, generic_info, generic_trace, generic_warn, GGMLLogLevel,
};
use core::ffi::{c_char, c_void};
use std::borrow::Cow;
use std::ffi::CStr;
use std::sync::Once;
use ggml_all_sys_2::ggml_log_level;

static WHISPER_LOG_TRAMPOLINE_INSTALL: Once = Once::new();
pub(crate) fn install_whisper_logging_hook() {
    WHISPER_LOG_TRAMPOLINE_INSTALL.call_once(|| unsafe {
        ggml_all_sys_2::whisper_log_set(Some(whisper_logging_trampoline), std::ptr::null_mut())
    });
}

unsafe extern "C" fn whisper_logging_trampoline(
    level: ggml_log_level,
    text: *const c_char,
    _: *mut c_void, // user_data
) {
    if text.is_null() {
        generic_error!("whisper_logging_trampoline: text is nullptr");
    }
    let level = GGMLLogLevel::from(level);

    // SAFETY: we must trust whisper.cpp that it will not pass us a string that does not satisfy
    // from_ptr's requirements.
    let log_str = unsafe { CStr::from_ptr(text) }.to_string_lossy();

    whisper_logging_trampoline_safe(level, log_str)
}

// this code essentially compiles down to a noop if neither feature is enabled
#[cfg_attr(
    not(any(feature = "log_backend", feature = "tracing_backend")),
    allow(unused_variables)
)]
fn whisper_logging_trampoline_safe(level: GGMLLogLevel, text: Cow<str>) {
    match level {
        GGMLLogLevel::None => {
            // no clue what to do here, trace it?
            generic_trace!("{}", text.trim());
        }
        GGMLLogLevel::Info => {
            generic_info!("{}", text.trim());
        }
        GGMLLogLevel::Warn => {
            generic_warn!("{}", text.trim());
        }
        GGMLLogLevel::Error => {
            generic_error!("{}", text.trim());
        }
        GGMLLogLevel::Debug => {
            generic_debug!("{}", text.trim());
        }
        GGMLLogLevel::Cont => {
            // this means continue previous log
            // storing state to do this is a massive pain so it's just a lot easier to not
            // plus as far as i can tell it's not actually *used* anywhere
            // whisper splits at 1024 chars and doesn't actually change the kind
            // so technically this is unused
            generic_trace!("{}", text.trim());
        }
        GGMLLogLevel::Unknown(level) => {
            generic_warn!(
                "whisper_logging_trampoline: unknown log level {}: message: {}",
                level,
                text.trim()
            );
        }
    }
}
