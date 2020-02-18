#!/usr/bin/env Rscript

library("optparse")
library("tidyverse")
library("munsell")
library("RcppCNPy")

## Files -- all per timestamp, 0 to and including 1 is episode
## thing_rew_mod.npy -- Reward Mod, float
## viols.npy -- constraint violations, binary
## action.npy -- integer valued
## done.npy -- episodes
## raw_reward.npy -- Reward get before modification, clipped
## reward.npy -- Reward after modification, clipped

## Episode Accumulation

## Rolling Window

## Window Function?

## Plot and Window

## Select Best Index

## Truncate Match Sequences

## Plot on page

## Paramaterized Plotting

## Handle Loading/Producing Files

## Argument Parsing
