static int CVE_2010_0437_PATCHED_ip6_dst_lookup_tail(struct sock *sk,
			       struct dst_entry **dst, struct flowi *fl)
{
	int err;
	struct net *net = sock_net(sk);

	if (*dst == NULL)
		*dst = ip6_route_output(net, sk, fl);

	if ((err = (*dst)->error))
		goto out_err_release;

	if (ipv6_addr_any(&fl->fl6_src)) {
		err = ipv6_dev_get_saddr(ip6_dst_idev(*dst)->dev,
					 &fl->fl6_dst,
					 sk ? inet6_sk(sk)->srcprefs : 0,
					 &fl->fl6_src);
		if (err)
			goto out_err_release;
	}

#ifdef CONFIG_IPV6_OPTIMISTIC_DAD
	/*
	 * Here if the dst entry we've looked up
	 * has a neighbour entry that is in the INCOMPLETE
	 * state and the src address from the flow is
	 * marked as OPTIMISTIC, we release the found
	 * dst entry and replace it instead with the
	 * dst entry of the nexthop router
	 */
	if ((*dst)->neighbour && !((*dst)->neighbour->nud_state & NUD_VALID)) {
		struct inet6_ifaddr *ifp;
		struct flowi fl_gw;
		int redirect;

		ifp = ipv6_get_ifaddr(net, &fl->fl6_src,
				      (*dst)->dev, 1);

		redirect = (ifp && ifp->flags & IFA_F_OPTIMISTIC);
		if (ifp)
			in6_ifa_put(ifp);

		if (redirect) {
			/*
			 * We need to get the dst entry for the
			 * default router instead
			 */
			dst_release(*dst);
			memcpy(&fl_gw, fl, sizeof(struct flowi));
			memset(&fl_gw.fl6_dst, 0, sizeof(struct in6_addr));
			*dst = ip6_route_output(net, sk, &fl_gw);
			if ((err = (*dst)->error))
				goto out_err_release;
		}
	}
#endif

	return 0;

out_err_release:
	if (err == -ENETUNREACH)
		IP6_INC_STATS_BH(NULL, IPSTATS_MIB_OUTNOROUTES);
	dst_release(*dst);
	*dst = NULL;
	return err;
}
