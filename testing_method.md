For testing I do this : set storage metadata with following : {
//   "path": "D://VisArc_Storage/storage_metadata.json"
"path": "/Volumes/WD_Rishi/Remote_1/storage_metadata.json"
} , then set config with {
    "image_quality": 0.3,
    "enable_visual_chat": true,
    "llm_mode": "server"
    ,"embedding_model": "Qwen3-Embedding-4B-Q4_K_M.gguf"

    ,"mmproj_model" : "gemma3_4b_q4_k_m_mmproj-F16.gguf"
}, then perform load-rag post with nothing in body. After that I can