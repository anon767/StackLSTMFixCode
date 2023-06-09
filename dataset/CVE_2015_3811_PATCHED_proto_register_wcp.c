void
CVE_2015_3811_PATCHED_proto_register_wcp(void)
{
    static hf_register_info hf[] = {
	{ &hf_wcp_cmd,
	  { "Command", "wcp.cmd", FT_UINT8, BASE_HEX, VALS(cmd_string), WCP_CMD,
	    "Compression Command", HFILL }},
	{ &hf_wcp_ext_cmd,
	  { "Extended Command", "wcp.ext_cmd", FT_UINT8, BASE_HEX, VALS(ext_cmd_string), WCP_EXT_CMD,
	    "Extended Compression Command", HFILL }},
	{ &hf_wcp_seq,
	  { "SEQ", "wcp.seq", FT_UINT16, BASE_HEX, NULL, WCP_SEQ,
	    "Sequence Number", HFILL }},
	{ &hf_wcp_chksum,
	  { "Checksum", "wcp.checksum", FT_UINT8, BASE_DEC, NULL, 0,
	    "Packet Checksum", HFILL }},
	{ &hf_wcp_tid,
	  { "TID", "wcp.tid", FT_UINT16, BASE_DEC, NULL, 0,
	    NULL, HFILL }},
	{ &hf_wcp_rev,
	  { "Revision", "wcp.rev", FT_UINT8, BASE_DEC, NULL, 0,
	    NULL, HFILL }},
	{ &hf_wcp_init,
	  { "Initiator", "wcp.init", FT_UINT8, BASE_DEC, NULL, 0,
	    NULL, HFILL }},
	{ &hf_wcp_seq_size,
	  { "Seq Size", "wcp.seq_size", FT_UINT8, BASE_DEC, NULL, 0,
	    "Sequence Size", HFILL }},
	{ &hf_wcp_alg_cnt,
	  { "Alg Count", "wcp.alg_cnt", FT_UINT8, BASE_DEC, NULL, 0,
	    "Algorithm Count", HFILL }},
	{ &hf_wcp_alg_a,
	  { "Alg 1", "wcp.alg1", FT_UINT8, BASE_DEC, NULL, 0,
	    "Algorithm #1", HFILL }},
	{ &hf_wcp_alg_b,
	  { "Alg 2", "wcp.alg2", FT_UINT8, BASE_DEC, NULL, 0,
	    "Algorithm #2", HFILL }},
	{ &hf_wcp_alg_c,
	  { "Alg 3", "wcp.alg3", FT_UINT8, BASE_DEC, NULL, 0,
	    "Algorithm #3", HFILL }},
	{ &hf_wcp_alg_d,
	  { "Alg 4", "wcp.alg4", FT_UINT8, BASE_DEC, NULL, 0,
	    "Algorithm #4", HFILL }},
	{ &hf_wcp_alg,
	  { "Alg", "wcp.alg", FT_UINT8, BASE_DEC, NULL, 0,
	    "Algorithm", HFILL }},
#if 0
	{ &hf_wcp_rexmit,
	  { "Rexmit", "wcp.rexmit", FT_UINT8, BASE_DEC, NULL, 0,
	    "Retransmit", HFILL }},
#endif
	{ &hf_wcp_hist_size,
	  { "History", "wcp.hist", FT_UINT8, BASE_DEC, NULL, 0,
	    "History Size", HFILL }},
	{ &hf_wcp_ppc,
	  { "PerPackComp", "wcp.ppc", FT_UINT8, BASE_DEC, NULL, 0,
	    "Per Packet Compression", HFILL }},
	{ &hf_wcp_pib,
	  { "PIB", "wcp.pib", FT_UINT8, BASE_DEC, NULL, 0,
	    NULL, HFILL }},
	{ &hf_wcp_compressed_data,
	  { "Compressed Data", "wcp.compressed_data", FT_NONE, BASE_NONE, NULL, 0,
	    "Raw compressed data", HFILL }},
	{ &hf_wcp_comp_bits,
	  { "Compress Flag", "wcp.flag", FT_UINT8, BASE_HEX, NULL, 0,
	    "Compressed byte flag", HFILL }},
#if 0
	{ &hf_wcp_comp_marker,
	  { "Compress Marker", "wcp.mark", FT_UINT8, BASE_DEC, NULL, 0,
	    "Compressed marker", HFILL }},
#endif
	{ &hf_wcp_offset,
	  { "Source offset", "wcp.off", FT_UINT16, BASE_HEX, NULL, WCP_OFFSET_MASK,
	    "Data source offset", HFILL }},
	{ &hf_wcp_short_len,
	  { "Compress Length", "wcp.short_len", FT_UINT8, BASE_HEX, NULL, 0xf0,
	    "Compressed length", HFILL }},
	{ &hf_wcp_long_len,
	  { "Compress Length", "wcp.long_len", FT_UINT8, BASE_HEX, NULL, 0,
	    "Compressed length", HFILL }},
	{ &hf_wcp_long_run,
	  { "Long Compression", "wcp.long_comp", FT_BYTES, BASE_NONE, NULL, 0,
	    "Long Compression type", HFILL }},
	{ &hf_wcp_short_run,
	  { "Short Compression", "wcp.short_comp", FT_BYTES, BASE_NONE, NULL, 0,
	    "Short Compression type", HFILL }},

   };


    static gint *ett[] = {
        &ett_wcp,
        &ett_wcp_comp_data,
	&ett_wcp_field,
    };

    static ei_register_info ei[] = {
        { &ei_wcp_compressed_data_exceeds, { "wcp.compressed_data.exceeds", PI_MALFORMED, PI_ERROR, "Compressed data exceeds maximum buffer length", EXPFILL }},
        { &ei_wcp_uncompressed_data_exceeds, { "wcp.uncompressed_data.exceeds", PI_MALFORMED, PI_ERROR, "Uncompressed data exceeds maximum buffer length", EXPFILL }},
               { &ei_wcp_invalid_window_offset, { "wcp.off.invalid", PI_MALFORMED, PI_ERROR, "Offset points outside of visible window", EXPFILL }},
               { &ei_wcp_invalid_match_length, { "wcp.len.invalid", PI_MALFORMED, PI_ERROR, "Length greater than offset", EXPFILL }},
    };

    expert_module_t* expert_wcp;

    proto_wcp = proto_register_protocol ("Wellfleet Compression", "WCP", "wcp");
    proto_register_field_array (proto_wcp, hf, array_length(hf));
    proto_register_subtree_array(ett, array_length(ett));
    expert_wcp = expert_register_protocol(proto_wcp);
    expert_register_field_array(expert_wcp, ei, array_length(ei));
}
